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

BATCH_SIZE=128

def affine_transform_xyr(b_x,dx,dy,dr):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine( translate_percent={"x": dx, "y": dy}, rotate=dr * 180 )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_60(b_x):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine( scale={"x": 0.65, "y": 0.65} )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

we0=np.load('/Users/fengqi/Pycharm_py36/QF/we0.npy')
we1=np.load('/Users/fengqi/Pycharm_py36/QF/we1.npy')
we2=np.load('/Users/fengqi/Pycharm_py36/QF/we2.npy')
we3=np.load('/Users/fengqi/Pycharm_py36/QF/we3.npy')
wem=np.load('/Users/fengqi/Pycharm_py36/QF/wem.npy')
wd0=np.load('/Users/fengqi/Pycharm_py36/QF/wd0.npy')
wd1=np.load('/Users/fengqi/Pycharm_py36/QF/wd1.npy')
wd2=np.load('/Users/fengqi/Pycharm_py36/QF/wd2.npy')
wd3=np.load('/Users/fengqi/Pycharm_py36/QF/wd3.npy')
wdd=np.load('/Users/fengqi/Pycharm_py36/QF/wdd.npy')

be0=np.load('/Users/fengqi/Pycharm_py36/QF/be0.npy')
be1=np.load('/Users/fengqi/Pycharm_py36/QF/be1.npy')
be2=np.load('/Users/fengqi/Pycharm_py36/QF/be2.npy')
be3=np.load('/Users/fengqi/Pycharm_py36/QF/be3.npy')
bem=np.load('/Users/fengqi/Pycharm_py36/QF/bem.npy')
bd0=np.load('/Users/fengqi/Pycharm_py36/QF/bd0.npy')
bd1=np.load('/Users/fengqi/Pycharm_py36/QF/bd1.npy')
bd2=np.load('/Users/fengqi/Pycharm_py36/QF/bd2.npy')
bd3=np.load('/Users/fengqi/Pycharm_py36/QF/bd3.npy')
bdd=np.load('/Users/fengqi/Pycharm_py36/QF/bdd.npy')


scale1 = 4
n_l0 = 64 * scale1;
n_l1 = 32 * scale1;
n_l2 = 16 * scale1;
n_l3 = 8 * scale1;
n_encoded = 32  # 4*scale1#pow(4,ii)
n_d0 = 8 * scale1;
n_d1 = 16 * scale1;
n_d2 = 32 * scale1;
n_d3 = 64 * scale1;
n_decoded = 784


tot_t_steps=10

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None,  28 * 28])  # value in the range of (0, 1)
ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
ph_switch = tf.placeholder(tf.float32, [1])
ph_lr = tf.placeholder(tf.float32, [])
ph_dis_e = tf.placeholder(tf.float32, [None, n_encoded])
# encoder


en0 = tf.layers.dense(tf_x, n_l0, tf.nn.sigmoid, kernel_initializer=tf.constant_initializer(we0),bias_initializer=tf.constant_initializer(be0),trainable=False)
en1 = tf.layers.dense(en0, n_l1, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(we1),bias_initializer=tf.constant_initializer(be1),trainable=False)
en2 = tf.layers.dense(en1, n_l2, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(we2),bias_initializer=tf.constant_initializer(be2),trainable=False)
en3 = tf.layers.dense(en2, n_l3, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(we3),bias_initializer=tf.constant_initializer(be3),trainable=False)

# RNN

TIME_STEP=10;INPUT_SIZE=32;CELL_SIZE=32;PFC_LearningRate=0.001;OUT_SIZE=32

PFC_x = tf.placeholder(tf.float32, [None, TIME_STEP-2, INPUT_SIZE])        # shape(batch, 5, 1)
PFC_y = tf.placeholder(tf.float32, [None, TIME_STEP-2,OUT_SIZE])

PFC_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
init_s = PFC_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
outs, final_s = tf.nn.dynamic_rnn(
                                        PFC_cell,                   # cell you have chosen
                                        PFC_x,                      # input
                                        initial_state=init_s,       # the initial hidden state
                                        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                                      )
# outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
# net_outs2D = tf.layers.dense(outs2D, OUT_SIZE)
# outs = tf.reshape(net_outs2D, [-1, TIME_STEP-2, OUT_SIZE])          # reshape back to 3D

PFC_loss = tf.losses.mean_squared_error(labels=PFC_y, predictions=outs)  # compute cost
PFC_train = tf.train.AdamOptimizer(learning_rate=PFC_LearningRate).minimize(PFC_loss)


#imagination
dcd_x=tf.placeholder(tf.float32,[None,OUT_SIZE])
# decoder
ff2 = tf.layers.dense(dcd_x, n_encoded, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wem),bias_initializer=tf.constant_initializer(bem),trainable=False)
de0 = tf.layers.dense(ff2, n_d0, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd0),bias_initializer=tf.constant_initializer(bd0),trainable=False)
de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd1),bias_initializer=tf.constant_initializer(bd1),trainable=False)
de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd2),bias_initializer=tf.constant_initializer(bd2),trainable=False)
de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd3),bias_initializer=tf.constant_initializer(bd3),trainable=False)
decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wdd),bias_initializer=tf.constant_initializer(bdd),trainable=False)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets('./mnist', one_hot=False)

tf.set_random_seed(1)



tot_episodes=10000
cost_his=np.zeros([tot_episodes])
LearningRate = 0.001;ph_lr_ = np.ones(shape=[]) * LearningRate


eps_num=10000

saver = tf.train.Saver()
file_addr='/Users/fengqi/Pycharm_py36/QF/RNN'+str(eps_num)+'/CKPT001'
saver.restore(sess, file_addr)


f, a = plt.subplots(2,10)
for ep in range (1):#episode
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x_s60=affine_transform_60(b_x)
    a0=-0.5
    a1=0.5
    num1=np.random.random([2])*(a1-a0)+a0
    x_itr=np.linspace(num1[0], num1[1], num=tot_t_steps)
    num1 = np.random.random([2]) * (a1 - a0) + a0
    y_itr = np.linspace(num1[0], num1[1], num=tot_t_steps)
    num1 = np.random.random([2]) * (a1 - a0) + a0
    r_itr = np.linspace(num1[0], num1[1], num=tot_t_steps)

    srsvrv_ = np.zeros([BATCH_SIZE, 4])

    for time_step in range(tot_t_steps):
        if time_step == 0:
            PFC_in  = np.zeros([BATCH_SIZE,tot_t_steps,INPUT_SIZE])

        im = affine_transform_xyr(b_x_s60, x_itr[time_step], y_itr[time_step], r_itr[time_step])
        a[0][time_step].imshow(np.reshape(im[2],[28,28]))
        srsvrv_[:, 0] =x_itr[time_step]
        srsvrv_[:, 1] =y_itr[time_step]
        srsvrv_[:, 2] =r_itr[time_step]
        srsvrv_[:, 3] =0
        PFC_in[:,time_step,:]=sess.run (en3, {tf_x: im})



    outs_=sess.run(outs,{PFC_x: PFC_in[:,1:tot_t_steps-1,:]})

    for time_step in range(tot_t_steps):
        if time_step > 1:
            im_dcd_=sess.run(decoded,{dcd_x:outs_[:,time_step-2,:]})
            a[1][time_step].imshow(np.reshape(im_dcd_[2], [28, 28]))
        else:
            a[1][time_step].imshow(np.zeros( [28, 28]))
file_nm='/Users/fengqi/Pycharm_py36/QF/_RNN'+str(eps_num)+str(np.random.random(1))+'.jpeg'

plt.savefig(file_nm)
