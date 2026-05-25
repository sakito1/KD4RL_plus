import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib import rnn

class lstm_drl(object):
    def __init__(self,inputs,keep_prob,trade_num,params):
        self.inputs = inputs
        self.keep_prob = keep_prob
        self.trade_num = trade_num

        self.params = params
        self.global_step =  tf.Variable(0, trainable=False)
        self.build_network()

    def unit_lstm(self,hidden_size,is_last=False):
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        if is_last:
            lstm_cell = rnn.AttentionCellWrapper(lstm_cell,len(self.inputs))
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        return lstm_cell



    def batch_attention_output(self,D):
        # attention_weight =  tf.matmul(tf.matmul(D,self.W_cor_output) ,D ,transpose_b=True)
        attention_dot = tf.matmul(D,D , transpose_b=True)
        QK = attention_dot / np.sqrt(float(self.params['hidden_size2']))
        self.attention_weight = tf.nn.softmax(QK)
        attention = tf.matmul(self.attention_weight,D)
        return attention



    def build_network(self):
        #2-layer lstm
        self.mlstm_cell = rnn.MultiRNNCell([self.unit_lstm(self.params['hidden_size1']),
                                            self.unit_lstm(self.params['hidden_size2'] ,is_last=False)],
                                            state_is_tuple=True)

        #fc layer - 1
        self.W_1 = tf.Variable(tf.truncated_normal([self.params['hidden_size2'],
                                                    self.params['hidden_size1']], stddev=0.01),
                                                    dtype=tf.float32)
        self.B_1 = tf.Variable(0.01, dtype=tf.float32)

        #fc layer - 2
        self.W_2 = tf.Variable(tf.truncated_normal([self.params['hidden_size1'], 1], stddev=0.01),
                                                    dtype=tf.float32)
        self.B_2 = tf.Variable(0.01, dtype=tf.float32)

        # self.W_cor = tf.Variable(tf.truncated_normal([7,7] ,stddev=0.01) , dtype=tf.float32)
        # self.W_cor_output = tf.Variable(tf.truncated_normal([self.params['hidden_size2'],self.params['hidden_size2']] ,stddev=0.01) , dtype=tf.float32)

        for index in range(len(self.inputs)):
            X = self.inputs[index]
            t_input = X[:,:-1,:]
            next_ret = X[:,-1,0] + 1
            # attention_input = self.batch_attention(t_input)
            outputs, state = tf.nn.dynamic_rnn(self.mlstm_cell, inputs=t_input, dtype=tf.float32, time_major=False)
            h_state = outputs[:,-1,:]

            #prior rank
            # rank =self.ranks[index]
            # rank_embed = tf.contrib.layers.embed_sequence(rank,vocab_size=self.params['lookup_size'],embed_dim=self.params['hidden_size1'])
            # W_psi = tf.Variable(tf.truncated_normal([self.params['hidden_size1'],1] ,stddev=0.01) , dtype=tf.float32)
            #
            # flattened = tf.nn.dropout(tf.matmul(tf.reshape(rank_embed,[-1,
            #                                                     self.params['hidden_size1']]) , W_psi),keep_prob=self.keep_prob)
            #
            #
            # # psi = tf.sigmoid(tf.einsum('ijk,k->ij',rank_embed, W_psi))
            # psi = tf.sigmoid(tf.reshape(flattened,[self.shapes[index],self.shapes[index]]))

            attn = self.batch_attention_output(h_state)

            fc1_out = tf.matmul(attn,self.W_1) + self.B_1
            fc1_act = tf.tanh(fc1_out)
            fc1_drop = tf.nn.dropout(fc1_act,self.keep_prob)
            fc2_out = tf.matmul(fc1_drop,self.W_2) + self.B_2
            fc2_act = tf.tanh(fc2_out)

            self.pred = tf.matrix_transpose(fc2_act)
            top_val, top_indices = tf.nn.top_k(self.pred, k=self.trade_num[index])
            down_val, down_indices = tf.nn.top_k( - self.pred, k=self.trade_num[index])

            buy_por = tf.nn.softmax(top_val, dim=1)[0, :]
            sell_por = tf.nn.softmax(down_val, dim=1)[0, :]

            portfolio_month_return = tf.reduce_sum(buy_por * tf.gather(next_ret,top_indices) -
                                                   sell_por* tf.gather(next_ret,down_indices))
            portfolio_month_return = tf.reshape(portfolio_month_return, [1])

            # self.gradients = tf.gradients(fc2_act,t_input)
            self.gradients = tf.reduce_mean(tf.gradients(fc2_act,t_input)[0] ,axis=0)

            if index == 0:
                buy_target = top_indices
                sell_target = down_indices
                self.portfolio_return = portfolio_month_return
                self.stock_score = tf.reshape(self.pred, [1, -1])

            else:
                buy_target = tf.concat([buy_target, top_indices], 0)
                sell_target = tf.concat([sell_target, down_indices], 0)
                self.portfolio_return = tf.concat([self.portfolio_return, portfolio_month_return], 0)
                self.stock_score = tf.concat([self.stock_score, tf.reshape(self.pred, [1, -1])], 0)


        mean ,var = tf.nn.moments(self.portfolio_return , axes= [0])
        std = tf.sqrt(var)

        self.loss = - (mean * tf.sqrt(12.0)) / std
        # self.loss = -mean*12






if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sess  = tf.Session(config=config)


    # inputs = [tf.placeholder(tf.float32) for i in range(5)]
    # keep_prob = tf.placeholder(tf.float32)
    # model = lstm_drl(inputs,keep_prob,{})
    # sess.run(tf.global_variables_initializer())
    # fd = {p:v for p,v in zip(inputs,range(1,6))}
    # fd.update({keep_prob:1.0})
    # print(sess.run(model.b,feed_dict=fd))