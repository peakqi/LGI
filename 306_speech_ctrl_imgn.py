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
XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
# total mv
def affine_transform1(b_x):
    xx = np.random.uniform(low=-XX, high=XX, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-XX, high=XX, size=1)
    sx = np.random.uniform(low=SX1, high=SX2, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=SX1, high=SX2, size=1)
    rr = np.random.uniform(low=-RX, high=RX, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy,sx,sy, rr), axis=0)

def affine_transform_xyr(b_x,dx,dy,dr):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine( translate_percent={"x": dx, "y": dy}, rotate=dr * 180 )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
def affine_transform_x(b_x,dx):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine( translate_percent={"x": dx})])
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
ff1 = tf.layers.dense(en3, n_encoded, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wem),bias_initializer=tf.constant_initializer(bem),trainable=False)

# RNN

TIME_STEP=20;INPUT_SIZE=70;CELL_SIZE=32;PFC_LearningRate=0.001;OUT_SIZE=CELL_SIZE

PFC_x = tf.placeholder(tf.float32, [None, TIME_STEP-1,INPUT_SIZE])        # shape(batch, 5, 1)
PFC_y = tf.placeholder(tf.float32, [None, TIME_STEP-1,OUT_SIZE])

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
dcd_in=tf.placeholder(tf.float32,[None,OUT_SIZE])
# decoder
ff2 = tf.layers.dense(dcd_in, n_encoded, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wem),bias_initializer=tf.constant_initializer(bem),trainable=False)
de0 = tf.layers.dense(ff2, n_d0, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd0),bias_initializer=tf.constant_initializer(bd0),trainable=False)
de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd1),bias_initializer=tf.constant_initializer(bd1),trainable=False)
de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd2),bias_initializer=tf.constant_initializer(bd2),trainable=False)
de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd3),bias_initializer=tf.constant_initializer(bd3),trainable=False)
decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wdd),bias_initializer=tf.constant_initializer(bdd),trainable=False)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets('./mnist', one_hot=False)

tf.set_random_seed(1)


tot_episodes=100000
cost_his=np.zeros([tot_episodes])
ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])
LearningRate = 0.001;ph_lr_ = np.ones(shape=[]) * LearningRate
PFC_x_=np.zeros([BATCH_SIZE,TIME_STEP,70])

spc_bi=char_to_bi(' ')
f,a=plt.subplots(2,TIME_STEP-1)
b_x_1=np.zeros([BATCH_SIZE,784])




for ep in range (tot_episodes):#episode
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    for ii in range(BATCH_SIZE):
        b_x_1[ii],para=affine_transform1(np.reshape(b_x[ii],[1,784]))


    a0=0;a1=0.2
    d1 = np.random.random() * (a1 - a0) + a0
    if np.random.random([1])>0.5:
        d2='left '
        d1=d1*-1
    else:
        d2='right '
    str_d1=str(d1)
    cmd='move '+d2+str_d1[1:6]+'$'
    cmd_sz1 = len(cmd)
    cmd_bi=sentence_to_vec(cmd)

    b_x_2=affine_transform_x(b_x_1,d1)
    en3_1,ff1_1= sess.run([en3,ff1],{tf_x: b_x_1, ph_encoded: ph_encoded_,ph_switch: ph_switch_,ph_dis_e: ph_dis_e_})
    en3_2,ff1_2 = sess.run([en3, ff1], {tf_x: b_x_2, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})

    cmd_sz=len(cmd)
    for jj in range(TIME_STEP):
        if jj<cmd_sz:
            aud = np.tile(cmd_bi[jj], [BATCH_SIZE, 1])
            PFC_x_[:, jj, :] = np.concatenate((en3_1, ff1_1,aud), axis=1)
        else:
            cmd=cmd+' '
            aud=np.tile(spc_bi, [BATCH_SIZE, 1])
            PFC_x_[:, jj, :] = np.concatenate(( en3_2, ff1_2,aud), axis=1)
    _,cost_his[ep]=sess.run([PFC_train,PFC_loss],{PFC_x: PFC_x_[:,0:TIME_STEP-1,:],PFC_y:PFC_x_[:,1:TIME_STEP,:32]})

    if (ep % 100 == 0):
        print('ep_step:', ep, '| train loss: %.8f' % cost_his[ep])
        np.save('cost', cost_his)




    if ((ep+1) % 1000 == 0):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        for ii in range(BATCH_SIZE):
            b_x_1[ii], para = affine_transform1(np.reshape(b_x[ii], [1, 784]))

        a0 = 0;
        a1 = 0.2
        d1 = 0.2
        if np.random.random([1]) > 0.5:
            d2 = 'left '
            d1 = d1 * -1
        else:
            d2 = 'right '
        str_d1 = str(d1)
        cmd = 'move ' + d2 + str_d1[1:6] + '$'
        cmd_sz1 = len(cmd)
        cmd_bi = sentence_to_vec(cmd)

        b_x_2 = affine_transform_x(b_x_1, d1)
        en3_1, ff1_1 = sess.run([en3, ff1],
                                {tf_x: b_x_1, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})
        en3_2, ff1_2 = sess.run([en3, ff1],
                                {tf_x: b_x_2, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})

        cmd_sz = len(cmd)
        for jj in range(TIME_STEP):
            if jj < cmd_sz:
                aud = np.tile(cmd_bi[jj], [BATCH_SIZE, 1])
                PFC_x_[:, jj, :] = np.concatenate((en3_1, ff1_1, aud), axis=1)
            else:
                cmd = cmd + ' '
                aud = np.tile(spc_bi, [BATCH_SIZE, 1])
                PFC_x_[:, jj, :] = np.concatenate((en3_2, ff1_2, aud), axis=1)


        view_im_ind = 0
        outs_ = sess.run(outs, {PFC_x: PFC_x_[:, 0:TIME_STEP - 1, :]})
        for kk in range(TIME_STEP-1):
            im_dcd_ = sess.run(decoded, {dcd_in: np.reshape(outs_[:, kk, :], [BATCH_SIZE, 32])})
            a[0][kk].set_xticks(());
            a[0][kk].set_yticks(())
            a[1][kk].set_xticks(());
            a[1][kk].set_yticks(())
            a[0][kk].set_title(cmd[kk])

            if kk < cmd_sz:
                a[0][kk].imshow(np.reshape(b_x_1[view_im_ind], [28, 28]))
            else:
                a[0][kk].imshow(np.reshape(b_x_2[view_im_ind], [28, 28]))

            a[1][kk].imshow(np.reshape(im_dcd_[view_im_ind], [28, 28]))
        file_nm = '/Users/fengqi/Pycharm_py36/QF/Lng_RNN' + str(ep) + str(np.random.random(1)) + '.jpeg'
        plt.savefig(file_nm)

        try:

            saver = tf.train.Saver()
            path = '/Users/fengqi/Pycharm_py36/QF/Lng_RNN' + str(ep+1) + '/'
            file_='CKPT001'
            os.mkdir(path)
            saver.save(sess, path+file_)
        except FileExistsError:
            aaa = 1





