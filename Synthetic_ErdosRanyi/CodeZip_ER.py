#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt 
import time
import numpy as np
import pickle
from tqdm.notebook import tqdm, trange
import random

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import *
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm, trange

import seaborn as sns

from random import shuffle
from multiprocessing import Pool
import multiprocessing
from functools import partial
from networkx.generators.classic import cycle_graph
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from deepwalk import OnlyWalk

import os, sys
import scipy






class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#### Graph Generations

def shuffle_two_lists(list_1, list_2):

    c = list(zip(list_1, list_2))
    random.shuffle(c) 
    return zip(*c)



from collections import deque




#%%

def generate_ER_graphs(n_min, n_max, num, p, all_connected):
  ER_graph_list = []
  for i in tqdm(range(num)):
    while(True):
      n = random.choice(np.arange(n_min,n_max,1))
      ER_graph = nx.generators.random_graphs.erdos_renyi_graph(n, p)
      if not all_connected:
          ER_graph.remove_nodes_from(list(nx.isolates(ER_graph)))
          ER_graph_list.append(ER_graph)
          break
      else:
          if nx.is_connected(ER_graph):
              ER_graph_list.append(ER_graph)
              break
          else:
              continue
  return ER_graph_list


def generate_positive_motifs(m):

    
  caveman_graph = nx.from_numpy_matrix(np.matrix([[0,1,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0], [1,1,0,0,0,1,0,0,1], [0,0,0,0,1,1,0,0,0], [0,0,0,1,0,1,0,0,0], [0,0,1,1,1,0,0,0,1], [0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,1,0,1], [0,0,1,0,0,1,1,1,0]]))
  cycle_graph = nx.cycle_graph(10)
  wheel_graph = nx.from_numpy_matrix( np.matrix([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                               [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                               [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                               [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                               [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                               [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                                               [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]]) )                                                  

  grid_graph = nx.grid_graph([3,3])

  ladder_graph = nx.ladder_graph(5)
  circularladder_graph = nx.from_numpy_matrix(np.matrix([[0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                                       [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                                       [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                                                       [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                                                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                                       [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                                                       [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                                                       [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                                                       [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                                                       [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]]))

  hypercube_graph = nx.from_numpy_matrix( np.matrix([[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                   [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                   [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                                   [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                                                   [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                                                   [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                                                   [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                                                   [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]]) )

  complete_graph = nx.from_numpy_matrix( np.matrix( [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                   [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                                   [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                                                   [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                                                   [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                                   [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                                                   [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]) )
                 
  lollipop_graph = nx.from_numpy_matrix( np.matrix([[0, 1, 1, 1, 1, 1, 1, 0, 0],
                                                    [1, 0, 1, 1, 1, 1, 1, 0, 0],
                                                    [1, 1, 0, 1, 1, 1, 1, 0, 0],
                                                    [1, 1, 1, 0, 1, 1, 1, 0, 0],
                                                    [1, 1, 1, 1, 0, 1, 1, 0, 0],
                                                    [1, 1, 1, 1, 1, 0, 1, 0, 0],
                                                    [1, 1, 1, 1, 1, 1, 0, 1, 0],
                                                    [0, 0, 0, 0, 0, 0, 1, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0, 1, 0]]))
 

  if m==0:
    nx.draw(caveman_graph)
    plt.show()
    return caveman_graph
  if m==1:
    nx.draw(cycle_graph)
    plt.show()
    return cycle_graph
  if m==2:
    nx.draw(wheel_graph)
    plt.show()      
    return wheel_graph
  if m==3:
    nx.draw(grid_graph)
    plt.show()      
    return grid_graph
  if m==4:
    nx.draw(ladder_graph)
    plt.show()      
    return ladder_graph
  if m==5:
    nx.draw(circularladder_graph)
    plt.show()      
    return circularladder_graph
  if m==6:
    nx.draw(lollipop_graph)
    plt.show()      
    return lollipop_graph
  if m==7:
    nx.draw(hypercube_graph)
    plt.show()      
    return hypercube_graph




def generate_negative_motifs(positive_motifs_list, num, all_connected):
  negative_motifs_list = []

  for i in tqdm(range(len(positive_motifs_list))):

    nb_nodes = len(positive_motifs_list[i].nodes())
    nb_edges = len(positive_motifs_list[i].edges())
    p = nb_edges / ((nb_nodes-1)*nb_nodes) * 2
    
    curr_motif_repeats_list = []
    for r in range(num):
      while(True):
        ER_negative = nx.generators.random_graphs.erdos_renyi_graph(nb_nodes, p)
        #print(len(ER_negative.edges()), nb_edges)
        if len(ER_negative.edges()) == nb_edges:
            if not all_connected:
                ER_negative.remove_nodes_from(list(nx.isolates(ER_negative)))
                curr_motif_repeats_list.append(ER_negative)
                break
            else:
                if nx.is_connected(ER_negative):
                    curr_motif_repeats_list.append(ER_negative)
                    break
    negative_motifs_list.append(curr_motif_repeats_list)

  return negative_motifs_list


def compose_two_graphs(g1, g2, ER_p, all_connected):

  g1_adj = nx.linalg.graphmatrix.adjacency_matrix(g1).todense()
  g2_adj = nx.linalg.graphmatrix.adjacency_matrix(g2).todense()

  g1_nb_nodes = len(g1.nodes())
  g2_nb_nodes = len(g2.nodes())

  #print(g1_nb_nodes, g2_nb_nodes)

  composed_adj = scipy.linalg.block_diag(g1_adj, g2_adj)

  while(True):
    binomial_edges = np.random.binomial(n=1, p = ER_p, size = (g1_nb_nodes, g2_nb_nodes))
    if not all_connected:
        composed_adj[:g1_nb_nodes,g1_nb_nodes:] = binomial_edges
        composed_adj[g1_nb_nodes:,:g1_nb_nodes] = np.transpose(binomial_edges)
        break
    else:    
        if binomial_edges.max()>0:
            composed_adj[:g1_nb_nodes,g1_nb_nodes:] = binomial_edges
            composed_adj[g1_nb_nodes:,:g1_nb_nodes] = np.transpose(binomial_edges)
            break
        else:
            continue
  return nx.from_numpy_matrix(composed_adj)






def generate_graphs_labels_ER(m, n_min, n_max, num_train, num_val, num_test, p_base_er = 0.05, p_combine = 0.05, all_connected1 = True, all_connected2 = True):

    print("train generation")
    train_ER_graph_list = generate_ER_graphs(n_min=n_min, n_max=n_max, num=num_train, p=p_base_er, all_connected = all_connected1)
    
    print("positive train")
    
    positive_motifs_list = [generate_positive_motifs(m)]
    nx.draw(generate_positive_motifs(m))
    plt.show()
    print("negative train")
    negative_motifs_list = generate_negative_motifs(positive_motifs_list=positive_motifs_list, num=num_train, all_connected = all_connected1)

    positive_train_graphs = []
    negative_train_graphs = []
    for i in tqdm(range(num_train)):
      positive_train_graphs.append( compose_two_graphs(positive_motifs_list[0], train_ER_graph_list[i], p_combine, all_connected = all_connected2) )
      negative_train_graphs.append( compose_two_graphs(negative_motifs_list[0][i], train_ER_graph_list[i], p_combine, all_connected = all_connected2) )
      
    print("val generation")
    val_ER_graph_list = generate_ER_graphs(n_min=n_min, n_max=n_max, num=num_val, p=p_base_er, all_connected = all_connected1)
    positive_motifs_list = [generate_positive_motifs(m)]
    negative_motifs_list = generate_negative_motifs(positive_motifs_list=positive_motifs_list, num=num_val, all_connected = all_connected1)

    positive_val_graphs = []
    negative_val_graphs = []
    for i in tqdm(range(num_val)):
      positive_val_graphs.append( compose_two_graphs(positive_motifs_list[0], val_ER_graph_list[i], p_combine, all_connected = all_connected2) )
      negative_val_graphs.append( compose_two_graphs(negative_motifs_list[0][i], val_ER_graph_list[i], p_combine, all_connected = all_connected2) )
      

    test_ER_graph_list = generate_ER_graphs(n_min=n_min, n_max=n_max, num=num_test, p=p_base_er, all_connected = all_connected1)
    positive_motifs_list = [generate_positive_motifs(m)]
    negative_motifs_list = generate_negative_motifs(positive_motifs_list=positive_motifs_list, num=num_test, all_connected = all_connected1)

    positive_test_graphs = []
    negative_test_graphs = []
    for i in tqdm(range(num_test)):
      positive_test_graphs.append( compose_two_graphs(positive_motifs_list[0], test_ER_graph_list[i], p_combine, all_connected = all_connected2) )
      negative_test_graphs.append( compose_two_graphs(negative_motifs_list[0][i], test_ER_graph_list[i], p_combine, all_connected = all_connected2) )
            

    all_train_graphs = positive_train_graphs + negative_train_graphs
    all_val_graphs = positive_val_graphs + negative_val_graphs
    all_test_graphs = positive_test_graphs + negative_test_graphs
    
    all_train_labels = list(np.ones(num_train)) + list(np.zeros(num_train))
    all_val_labels = list(np.ones(num_val)) + list(np.zeros(num_val))
    all_test_labels = list(np.ones(num_test)) + list(np.zeros(num_test))
    
    
    
    all_train_graphs_shuffled, all_train_labels_shuffled = \
                          shuffle_two_lists(all_train_graphs, all_train_labels)
    all_val_graphs_shuffled, all_val_labels_shuffled = \
                          shuffle_two_lists(all_val_graphs, all_val_labels)
    all_test_graphs_shuffled, all_test_labels_shuffled = \
                         shuffle_two_lists(all_test_graphs, all_test_labels)
    
    all_train_graphs_shuffled = list(all_train_graphs_shuffled)
    all_train_labels_shuffled = list(all_train_labels_shuffled)
    all_val_graphs_shuffled = list(all_val_graphs_shuffled)
    all_val_labels_shuffled = list(all_val_labels_shuffled)
    all_test_graphs_shuffled = list(all_test_graphs_shuffled)
    all_test_labels_shuffled = list(all_test_labels_shuffled)


    return all_train_graphs_shuffled, all_train_labels_shuffled,\
           all_val_graphs_shuffled, all_val_labels_shuffled,\
           all_test_graphs_shuffled, all_test_labels_shuffled






#%%





def networkx_to_dgl_graphs(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled):

    for i in range(len(all_train_graphs_shuffled)):
        all_train_graphs_shuffled[i] = dgl.from_networkx(all_train_graphs_shuffled[i])

    for i in range(len(all_val_graphs_shuffled)):
        all_val_graphs_shuffled[i] = dgl.from_networkx(all_val_graphs_shuffled[i]) 

    for i in range(len(all_test_graphs_shuffled)):
        all_test_graphs_shuffled[i] = dgl.from_networkx(all_test_graphs_shuffled[i])

    return all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled
    


def dgl_to_networkx_graphs(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled):

    for i in range(len(all_train_graphs_shuffled)):
        all_train_graphs_shuffled[i] = nx.Graph(all_train_graphs_shuffled[i].to_networkx())

    for i in range(len(all_val_graphs_shuffled)):
        all_val_graphs_shuffled[i] = nx.Graph(all_val_graphs_shuffled[i].to_networkx())

    for i in range(len(all_test_graphs_shuffled)):
        all_test_graphs_shuffled[i] = nx.Graph(all_test_graphs_shuffled[i].to_networkx())

    return all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled
    



#%%
##### Generate masking

def generate_masking_GAT(train_graphs, val_graphs, test_graphs):

  train_masking = []
  val_masking = []
  test_masking = []

  print('Start generating GAT masking')

  for graph in train_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      train_masking.append(torch.from_numpy(adj))
  
  for graph in val_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      val_masking.append(torch.from_numpy(adj))

  for graph in test_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      test_masking.append(torch.from_numpy(adj))

  return train_masking, val_masking, test_masking






def MinMaxScaler(data):
    diff = data.transpose(0,1) - torch.min(data, axis = 1)[0]
    range = torch.max(data, axis = 1)[0] - torch.min(data, axis = 1)[0]
    return (diff / (range + 1e-7)).transpose(0,1)





    
    
