import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
from sklearn.manifold import TSNE
from matplotlib import cm

import scipy as sp
from sklearn.svm import SVR
import scipy

def char_to_bi(x):
    l=' 0123456789abcdefghijklmnopqrstuvwxyz.+-*~!@$%^&()'
    for i5 in range(2):
        for i4 in range(2):
            for i3 in range(2):
                for i2 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            ind=i0+i1*2+i2*4+i3*8+i4*16+i5*32
                            if l[ind]==x:
                                return np.transpose([i5,i4,i3,i2,i1,i0])

def sentence_to_vec(X):
    sz=len(X)
    y=np.zeros([sz,6])
    for ii in range(sz):
        y[ii,:]=char_to_bi(X[ii])
    return y
Full_len=30
cmd_posi=25
Aud_sz=6
def Gen_lng():

    d1 = np.random.randint(28)
    d3=d1
    if np.random.random([1]) > 0.5:
        d2 = 'left '
        d3 = d3 * -1
    else:
        d2 = 'right '
    str_d1 = str(d1)
    cmd = 'move ' + d2 + str_d1
    cmd_sz1 = len(cmd)
    for ii in range(Full_len-cmd_sz1):
        if len(cmd)==cmd_posi:
            cmd=cmd+'$'
        else:
            cmd=cmd+' '
    return cmd,d3


# LSTM

TIME_STEP=30;INPUT_SIZE=6;CELL_SIZE=10;PFC_LearningRate=0.001;OUT_SIZE=1;BATCH_SIZE=128;

IPS_x = tf.placeholder(tf.float32, [None, TIME_STEP,INPUT_SIZE])        # shape(batch, 30, 6)
IPS_y = tf.placeholder(tf.float32, [None, TIME_STEP,OUT_SIZE])          # shape(batch, 30, 1)

IPS_cell = tf.contrib.rnn.BasicLSTMCell(num_units=CELL_SIZE)
init_s = IPS_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
IPS_outputs, IPS_final_s = tf.nn.dynamic_rnn(
                                        IPS_cell,                   # cell you have chosen
                                        IPS_x,                      # input
                                        initial_state=init_s,       # the initial hidden state
                                        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                                      )
IPS_outs2D = tf.reshape(IPS_outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
IPS_net_outs2D = tf.layers.dense(IPS_outs2D, OUT_SIZE)
IPS_outs = tf.reshape(IPS_net_outs2D, [-1, TIME_STEP, OUT_SIZE])          # reshape back to 3D

IPS_loss = tf.losses.mean_squared_error(labels=IPS_y, predictions=IPS_outs)  # compute cost
IPS_train = tf.train.AdamOptimizer(learning_rate=PFC_LearningRate).minimize(IPS_loss)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
path = '/Users/fengqi/Pycharm_py36/QF/IPS' + str(100000) + '/'
file_ = 'CKPT001'
saver.restore(sess, path + file_)


aa=tf.global_variables()
IPS_lstm_kernel=sess.run(aa[0])
np.save('IPS_lstm_kernel.npy',IPS_lstm_kernel)
IPS_lstm_bias=sess.run(aa[1])
np.save('IPS_lstm_bias.npy',IPS_lstm_bias)
IPS_dense_kernel=sess.run(aa[2])
np.save('IPS_dense_kernel.npy',IPS_dense_kernel)
IPS_dense_bias=sess.run(aa[3])
np.save('IPS_dense_bias.npy',IPS_dense_bias)



