import tensorflow as tf
import numpy as np


class attn_head(tf.keras.layers.Layer):
    def __init__(self,hidden_dim, nb_nodes = None,in_drop=0.0, coef_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(attn_head,self).__init__() 

        self.activation = activation
        self.residual = residual
    
        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)  

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
        self.conv_no_bias2 = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
    
        self.conv_f1 = tf.keras.layers.Conv1D(1,1)
        self.conv_f2 = tf.keras.layers.Conv1D(1,1)

        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))
    
    

    def __call__(self,  seq,  bias_mat, training, msk_in, args):
               
        dot_product_similarity = args.dot_product_similarity
        concat_similarity = args.concat_similarity
    
        if dot_product_similarity:
            seq = self.in_dropout(seq,training = training)
            seq_fts = self.conv_no_bias(seq)
            seq_fts2 = self.conv_no_bias2(seq)
        
            logits = tf.matmul(seq_fts, tf.transpose(seq_fts2,[0,2,1]) ) / np.sqrt(seq_fts.shape[-1])


        elif concat_similarity:
            seq = self.in_dropout(seq,training = training)
            seq_fts = self.conv_no_bias(seq)
          
            f_1 = self.conv_f1(seq_fts)
            f_2 = self.conv_f2(seq_fts)     

            logits = tf.nn.leaky_relu(f_1 + tf.transpose(f_2,[0,2,1]))


        coefs = tf.exp(logits) * bias_mat
        coefs /= tf.transpose(tf.reduce_sum(coefs,axis=-1)[tf.newaxis]+1e-19, [0,2,1])
        coefs = self.coef_dropout(coefs,training = training)
    
        seq_fts = self.in_dropout(seq_fts,training = training)
        
        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        ret = vals + self.bias_zero

        return self.activation(ret) #, coefs, p_mat







class attn_head_sparse(tf.keras.layers.Layer):
    def __init__(self,hidden_dim, nb_nodes = None,in_drop=0.0, coef_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(attn_head_sparse,self).__init__()        
        
        self.activation = activation
        self.residual = residual
        
        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)        
        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
        self.conv_no_bias2 = tf.keras.layers.Conv1D(hidden_dim,1,use_bias=False)
        
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.hidden_dim = hidden_dim
        
        self.conv_f1 = tf.keras.layers.Conv1D(1,1)
        self.conv_f2 = tf.keras.layers.Conv1D(1,1)
        
        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim,1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))
        
    
        
    def __call__(self, seq,  bias_mat, training, msk_in):
         
        dot_product_similarity = args.dot_product_similarity
        concat_similarity = args.concat_similarity

        if dot_product_similarity:
            seq = self.in_dropout(seq,training = training)
            seq_fts1 = self.conv_no_bias(seq)
            seq_fts2 = self.conv_no_bias2(seq)
                 
            logits = (tf.matmul(seq_fts1, tf.transpose(seq_fts2,[0,2,1]) ) ) / np.sqrt(seq_fts.shape[-1])
            
            logits = tf.SparseTensor(bias_mat.indices, tf.gather_nd(logits, bias_mat.indices) * bias_mat.values, bias_mat.dense_shape)
            
            coefs = tf.sparse.softmax(logits)
            
           
        elif concat_similarity:
       
            seq = self.in_dropout(seq,training = training)
            seq_fts = self.conv_no_bias(seq)
     
            f_1 = self.conv_f1(seq_fts)
            f_2 = self.conv_f2(seq_fts)     
    
            f_1 = tf.reshape(f_1, (nb_nodes, 1))
            f_2 = tf.reshape(f_2, (nb_nodes, 1))
    
            f_1 = bias_mat * f_1
            f_2 = bias_mat * tf.transpose(f_2, [1,0])
    
            logits = tf.sparse.add(f_1, f_2)
                
            lrelu = tf.SparseTensor(indices=logits.indices, values=tf.nn.leaky_relu(logits.values), 
                        dense_shape=logits.dense_shape)
    
            coefs = tf.sparse.softmax(lrelu)
        
        coefs = tf.SparseTensor(indices=coefs.indices, values=tf.nn.dropout(coefs.values, 1.0 - self.coef_drop),
                            dense_shape=coefs.dense_shape)
        
        
        seq_fts = self.in_dropout(seq_fts,training = training)
        
        coefs = tf.sparse.reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, self.hidden_dim])
        
        ret = vals + self.bias_zero
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)                
            else:
                ret = ret + seq
                
        return self.activation(ret) 
