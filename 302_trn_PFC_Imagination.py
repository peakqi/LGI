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
LearningRate = 0.001
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
ff_encoded = tf.layers.dense(en3, n_encoded, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wem),bias_initializer=tf.constant_initializer(bem),trainable=False)
enc = ff_encoded * ph_switch + ph_encoded * (1 - ph_switch)
encoded = tf.multiply(enc, ph_dis_e)

PFC_IN=tf.concat([en3,ff_encoded],axis=1)
# weights_en0 = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')

# RNN

TIME_STEP=10;INPUT_SIZE=64;CELL_SIZE=64;PFC_LearningRate=0.01;OUT_SIZE=32

PFC_x = tf.placeholder(tf.float32, [None, TIME_STEP-2, INPUT_SIZE])        # shape(batch, 5, 1)
PFC_y = tf.placeholder(tf.float32, [None, TIME_STEP-2,OUT_SIZE])

PFC_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
init_s = PFC_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.nn.dynamic_rnn(
                                        PFC_cell,                   # cell you have chosen
                                        PFC_x,                      # input
                                        initial_state=init_s,       # the initial hidden state
                                        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                                      )
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
net_outs2D = tf.layers.dense(outs2D, OUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP-2, OUT_SIZE])          # reshape back to 3D

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
ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])
LearningRate = 0.001;ph_lr_ = np.ones(shape=[]) * LearningRate


for ep in range (tot_episodes):#episode
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
        # a[0][time_step].imshow(np.reshape(im[2],[28,28]))
        srsvrv_[:, 0] =x_itr[time_step]
        srsvrv_[:, 1] =y_itr[time_step]
        srsvrv_[:, 2] =r_itr[time_step]
        srsvrv_[:, 3] =0
        PFC_in[:,time_step,:]=sess.run (PFC_IN, {tf_x: im, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_, ph_dis_e: ph_dis_e_})



    _,cost_his[ep]=sess.run([PFC_train,PFC_loss],{PFC_x: PFC_in[:,1:tot_t_steps-1,:],PFC_y:PFC_in[:,2:tot_t_steps,0:32]})

    if (ep % 10 == 0):
        print('ep_step:', ep, '| train loss: %.6f' % cost_his[ep])
        np.save('cost', cost_his)

saver = tf.train.Saver()
saver.save(sess, '/Users/fengqi/Pycharm_py36/QF/_PFC')


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
        PFC_in[:,time_step,:]=sess.run (PFC_IN, {tf_x: im, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_, ph_dis_e: ph_dis_e_})



    outs_=sess.run(outs,{PFC_x: PFC_in[:,1:tot_t_steps-1,:]})

    for time_step in range(tot_t_steps):
        if time_step > 1:
            im_dcd_=sess.run(decoded,{dcd_x:outs_[time_step-2]})
            a[1][time_step].imshow(np.reshape(im_dcd_[2], [28, 28]))


plt.savefig('/Users/fengqi/Pycharm_py36/QF/_PFC.jpeg')

## imagination


#
#
#
#
#
# # decoder
# de0 = tf.layers.dense(encoded, n_d0, tf.nn.sigmoid)
# de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid)
# de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid)
# de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid)
# decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid)
#
# loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
# train = tf.train.AdamOptimizer(ph_lr).minimize(loss)
#
# weights_en0 = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')
# weights_en1 = tf.get_default_graph().get_tensor_by_name(os.path.split(en1.name)[0] + '/kernel:0')
# weights_en2 = tf.get_default_graph().get_tensor_by_name(os.path.split(en2.name)[0] + '/kernel:0')
# weights_en3 = tf.get_default_graph().get_tensor_by_name(os.path.split(en3.name)[0] + '/kernel:0')
# weights_mid = tf.get_default_graph().get_tensor_by_name(os.path.split(ff_encoded.name)[0] + '/kernel:0')
# weights_de0 = tf.get_default_graph().get_tensor_by_name(os.path.split(de0.name)[0] + '/kernel:0')
# weights_de1 = tf.get_default_graph().get_tensor_by_name(os.path.split(de1.name)[0] + '/kernel:0')
# weights_de2 = tf.get_default_graph().get_tensor_by_name(os.path.split(de2.name)[0] + '/kernel:0')
# weights_de3 = tf.get_default_graph().get_tensor_by_name(os.path.split(de3.name)[0] + '/kernel:0')
# weights_ddr = tf.get_default_graph().get_tensor_by_name(os.path.split(decoded.name)[0] + '/kernel:0')
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
#
# init_new_vars_op = tf.initialize_variables([weights_en0, v_7, v_8])
# sess.run(init_new_vars_op)
#
#
#
# saver = tf.train.Saver()
# type ='/Users/fengqi/Pycharm_py36/QF/4900000/' + '_n_32-batch128-lr0.001-0.2-0.5-0.7-0.2-nx0'
# saver.restore(sess, type)
#
# test_x=np.zeros([BATCH_SIZE,784])
#
#
# VIEW_SIZE=128
# ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
# ph_switch_ = np.ones(shape=[1])
# ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
#
# view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
#     [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
#      weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
#      de2, de3, decoded],
#     {tf_x: b_x,
#      ph_encoded: ph_encoded_,
#      ph_switch: ph_switch_,
#      ph_dis_e: ph_dis_e_})
#
# np.save('we0.npy',we0)
# np.save('we1.npy',we1)
# np.save('we2.npy',we2)
# np.save('we3.npy',we3)
# np.save('wem.npy',wem)
# np.save('wd1.npy',wd1)
# np.save('wd2.npy',wd2)
# np.save('wd3.npy',wd3)
# np.save('wdd.npy',wdd)




#
#
# TIME_STEP=10;INPUT_SIZE=32;CELL_SIZE=32;PFC_LearningRate=0.01
#
# PFC_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 5, 1)
# PFC_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])          # input y
#
# # RNN
# PFC_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
# init_s = PFC_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
# outputs, final_s = tf.nn.dynamic_rnn(
#     PFC_cell,                   # cell you have chosen
#     PFC_x,                       # input
#     initial_state=init_s,       # the initial hidden state
#     time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
# )
# outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
# net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
# outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])          # reshape back to 3D
#
# PFC_loss = tf.losses.mean_squared_error(labels=PFC_y, predictions=outs)  # compute cost
# PFC_train = tf.train.AdamOptimizer(learning_rate=PFC_LearningRate).minimize(PFC_loss)
#
#
#
#


#
#
# tf.train.init_from_checkpoint(type, {"/dense/Sigmoid:0": "dense/Sigmoid:0"})
#
# with tf.variable_scope("my_scope"):
#   print(tf.get_variable_scope().name)






#
# lnstep_list=[0,10000,14000,20000,100000,600000,1400000,2500000,4900000]
# col_sz=len(lnstep_list)
#
# f,a=plt.subplots(16,16)
# im=test_x[0,:]
# mm=2
# for kk in range(col_sz):
#     print(kk)
#     lnstep=lnstep_list[kk]
#     saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/_ckpt/' + str(lnstep) + '/' + type)
#     we0_ = sess.run(weights_en0,
#                     {tf_x: np.reshape(im, [1, 784]), ph_encoded: ph_encoded_, ph_switch: ph_switch_,ph_dis_e: ph_dis_e_})
#     tt=0
#     # aa = np.average(np.abs(we0_));
#     # aaa = 'step' + str(lnstep) + 'avg=' + str(aa)
#     # print(aaa)
#
#     for ii in range(16):
#         for jj in range(16):
#             a[jj][ii].set_xticks(());
#             a[jj][ii].set_yticks(());
#             W=np.reshape(we0_[:,tt], [28, 28])
#             W[0][0] = mm;W[27][27] = -1 * mm
#             W[W > mm] = mm;W[W < -1*mm] = -1 * mm;
#             a[jj][ii].imshow(W,cmap='bwr');tt=tt+1
#     # plt.subplots_adjust(left=0.4,right=1,bottom=0,top=1,wspace=0,hspace=0)
#     sttring = 'step' + str(lnstep)
#     f.suptitle(sttring)
#     plt.savefig('_sup_fig1_W/a_W'+str(lnstep)+'thr'+str(mm)+'.png')
#
# aaa=2
#
#
# lnstep_list=[0,10000,14000,20000,100000,600000,1400000,2500000,4900000]
# col_sz=len(lnstep_list)
#
# f,a=plt.subplots(16,16)
# im=test_x[0,:]
# mm=2
# for kk in range(col_sz):
#     print(kk)
#     lnstep=lnstep_list[kk]
#     saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/_ckpt/' + str(lnstep) + '/' + type)
#     we0_ = sess.run(weights_en0,
#                     {tf_x: np.reshape(im, [1, 784]), ph_encoded: ph_encoded_, ph_switch: ph_switch_,ph_dis_e: ph_dis_e_})
#     tt=0
#     # aa = np.average(np.abs(we0_));
#     # aaa = 'step' + str(lnstep) + 'avg=' + str(aa)
#     # print(aaa)
#
#     for ii in range(16):
#         for jj in range(16):
#             a[jj][ii].set_xticks(());
#             a[jj][ii].set_yticks(());
#             W=np.reshape(we0_[:,tt], [28, 28])
#             W[0][0] = mm;W[27][27] = -1 * mm
#             W[W > 0] = 1;W[W < 0] = -1 ;
#             a[jj][ii].imshow(W,cmap='bwr');tt=tt+1
#     # plt.subplots_adjust(left=0.4,right=1,bottom=0,top=1,wspace=0,hspace=0)
#     sttring='step'+str(lnstep)
#     f.suptitle(sttring)
#     plt.savefig('_sup_fig1_W/b_BlackWhite_W'+str(lnstep)+'thr'+str(mm)+'.png')
#
#
#
#
#
# aaa=2