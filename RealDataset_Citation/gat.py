import tensorflow as tf
from attn_head import * 


class inference(tf.keras.layers.Layer):
    def __init__(self,n_heads,hid_units,nb_classes, nb_nodes,Sparse,ffd_drop=0.0, attn_drop=0.0,activation = tf.nn.elu,residual = False):        
        super(inference,self).__init__()
        if not Sparse:
            attned_head = attn_head
        else:
            attned_head = attn_head_sparse
        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]
        
        if len(hid_units)>0:
            for i in range(n_heads[0]):
                self.attns.append(attned_head(hidden_dim = hid_units[0], nb_nodes = nb_nodes, in_drop = ffd_drop, coef_drop = attn_drop, activation = activation, residual = residual))
        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):                
                sec_attns.append(attned_head(hidden_dim = hid_units[i], nb_nodes = nb_nodes, in_drop = ffd_drop, coef_drop = attn_drop, activation = activation, residual = residual))
                self.sec_attns.append(sec_attns)
                
        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim = nb_classes, nb_nodes = nb_nodes, in_drop = ffd_drop, coef_drop = attn_drop, activation = lambda x: x, residual = residual))                

    
    def __call__(self,  inputs,  bias_mat, training, msk_in, args):        
        first_attn = []
        out = []
        if len(self.attns)>0:
            for indiv_attn in self.attns:
                first_attn.append(indiv_attn( seq = inputs,  bias_mat = bias_mat,training = training, msk_in = msk_in, args = args))
            h_1 = tf.concat(first_attn,axis = -1)     
        for sec_attns in self.sec_attns:
            next_attn = []
            for indiv_attns in sec_attns:
                next_attn.append(indiv_attn( seq = h_1,  bias_mat = bias_mat,training = training, msk_in = msk_in, args = args))
            h_1 = tf.concat(next_attn,axis = -1)
        for indiv_attn in self.final_attns:
            if len(self.attns)>0:
                 out.append(indiv_attn( seq=h_1,  bias_mat = bias_mat,training = training, msk_in = msk_in, args = args))
            else:
                 out.append(indiv_attn( seq=inputs,  bias_mat = bias_mat,training = training, msk_in = msk_in, args = args))
            
        logits = tf.add_n(out)/self.final_sum
        return logits
  




    
def train(model,inputs,  bias_mat,lbl_in,msk_in,training, args):        
    with tf.GradientTape() as tape:                
        logits,accuracy,loss = model( inputs = inputs,  training =True, bias_mat = bias_mat, lbl_in =  lbl_in, msk_in =  msk_in, args = args)             

    gradients = tape.gradient(loss,model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    args.optimizer.apply_gradients(gradient_variables)        
                
    return logits,accuracy,loss





def evaluate( model,inputs,  bias_mat,lbl_in,msk_in,training, args):                                                        
    logits,accuracy,loss = model(inputs= inputs,  bias_mat = bias_mat, lbl_in = lbl_in, msk_in = msk_in, training = False, args = args)                        
    return logits,accuracy,loss





class GAT(tf.keras.Model):
    def __init__(self, hid_units,n_heads, nb_classes, nb_nodes,Sparse,ffd_drop = 0.0,attn_drop = 0.0,activation = tf.nn.elu,residual=False):    
        super(GAT,self).__init__()
                 
        self.hid_units = hid_units         #[8]
        self.n_heads = n_heads             #[8,1]
        self.nb_classes = nb_classes
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual        
        self.inferencing = inference(n_heads,hid_units,nb_classes,nb_nodes,Sparse = Sparse,ffd_drop = ffd_drop,attn_drop = attn_drop, activation = activation,residual = residual)
        
    def masked_softmax_cross_entropy(self,logits, labels, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self,logits, labels, mask):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    
    def __call__(self,  inputs,  training,bias_mat,lbl_in,msk_in, args):     
        logits = self.inferencing(inputs = inputs,  bias_mat = bias_mat,training = training, msk_in = msk_in, args = args)        
        
        log_resh = tf.reshape(logits, [-1, self.nb_classes])        
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])        
        
        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * args.l2_coef
        loss = loss + lossL2
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)
        
        return logits,accuracy,loss    
    
    