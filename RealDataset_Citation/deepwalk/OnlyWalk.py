#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import random
import numpy as np
import multiprocessing
import networkx as nx


class Graph(object):
    def __init__(self, G):
        self.G = G
        self.is_directed = nx.is_directed(G)
        self.add_weight()
        

    def add_weight(self):
        for e in self.G.edges:
              src = e[0]
              dst = e[1]
              self.G[src][dst]['weight'] = 1.0          
              if not self.is_directed:
                  self.G[dst][src]['weight'] = 1.0
                  
                  
    def update_weight(self, transP):
        for e in self.G.edges:
            src = e[0]
            dst = e[1]
            self.G[src][dst]['weight'] = transP[src][dst]
            if not self.is_directed:
                self.G[dst][src]['weight'] = transP[dst][src]
            
            


class Node2vec_onlywalk(object):

    def __init__(self, graph, path_length, num_paths, p=1.0, q=1.0, stop_prob = 0.0, with_freq_mat = True, **kwargs):

        
        kwargs["workers"] = kwargs.get("workers", 1)

       
        self.graph = Graph(graph)
        
        
        
        self.walker = Walker_onlywalk( self.graph, p=p, q=q, with_freq_mat = with_freq_mat, stop_prob = stop_prob, workers=kwargs["workers"])
        self.walker.preprocess_transition_probs()
        
        sentences = self.walker.simulate_walks( num_walks=num_paths, walk_length=path_length)
        #kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["sg"] = 1

        #self.size = kwargs["size"]
        self.sentences = sentences
        

        
        

class Walker_onlywalk:
    def __init__(self, G, p, q, with_freq_mat, workers, stop_prob):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = self.G.number_of_nodes()
        self.walks_dict = dict()
        self.stop_prob = stop_prob
        self.with_freq_mat = with_freq_mat
        if with_freq_mat:
            self.freq_mat = np.zeros([self.node_size, self.node_size])

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        
        walk = [start_node]
        
        if self.with_freq_mat:
            self.freq_mat[int(start_node), int(start_node)] +=1

        while len(walk) < walk_length:
            
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                
                if np.random.choice([0, 1], p = [self.stop_prob, 1 - self.stop_prob]):
                    
                    if len(walk) == 1:
                        walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                        
                    else:
                        prev = walk[-2]
                        pos = (prev, cur)
                        next = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                        walk.append(next)
                    if self.with_freq_mat:
                        self.freq_mat[int(start_node), int(walk[-1])] += 1
                    #self.freq_mat[int(start_node), int(walk[-1])] += 1
                else:
                    #print("early stop")
                    break
                
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        #print('Walk iteration:')
        for walk_iter in range(num_walks):
            #print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))
                if walks[-1][0] in self.walks_dict:
                    self.walks_dict[walks[-1][0]].append(walks[-1])
                else:
                    self.walks_dict[walks[-1][0]] = []
                    self.walks_dict[walks[-1][0]].append(walks[-1])

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[(edge[0],edge[1])] = self.get_alias_edge(edge[0], edge[1])
            if G.is_directed:
                alias_edges[(edge[1],edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return



def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
    
    
    
    
    
    

grf = nx.generators.trees.random_tree(50)
#graph = Graph(grf)

n2v = Node2vec_onlywalk(graph = grf, path_length=10, num_paths=50, p=1e6, q=1.0, stop_prob = 0.0, with_freq_mat = True)









