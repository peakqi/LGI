import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os

def affine_transform_x(b_x,dx):
    try:
        sz1, sz2 = b_x.shape
    except:
        sz1=1; sz2=784
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine( translate_percent={"x": dx/56})])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
def affine_transform_s(b_x,s):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine( scale={"x": s, "y": s})])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
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

def affine_transform_r(b_x,rr):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x


def affine_transform60(b_x):
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(scale={"x": 0.6, "y": 0.6} )])
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
def aud_code_to_txt(code):
    l = ' 0123456789abcdefghijklmnopqrstuvwxyz.+-*~!@$%^&()'
    ind = np.int(code[5] +  code[4]* 2 + code[3] * 4 + code[2] * 8 + code[1] * 16 + code[0] * 32)
    if ind<0 or ind>50:
        ind=0
    return l[ind]
def sentence_to_vec(X):
    sz=len(X)
    y=np.zeros([sz,6])
    for ii in range(sz):
        y[ii,:]=char_to_bi(X[ii])
    return y
Full_len=30
Aud_sz=6

def Gen_lng_move():
    d1 = 8
    d3=d1
    LR=0
    #train move left/right 28
    if np.random.random([1]) > 0.5:
        d2 = 'left '
        d3 = d3 * -1
        LR=-1
    else:
        d2 = 'right '
        LR=1
    str_d1 = str(d1)
    cmd = 'move ' + d2 + str_d1
    cmd_sz1 = len(cmd)
    for ii in range(Full_len-cmd_sz1):
       cmd=cmd+' '
    return cmd,d3,LR,cmd_sz1

def Gen_lng_rott():
    # a=180#[-30,-15,15,30]
    # d1 = np.random.randint(4)
    # d2 = a[d1]
    d2=180
    str_d = str(d2)
    cmd = 'rotate ' +  str_d
    cmd_sz1 = len(cmd)
    for ii in range(Full_len-cmd_sz1):
       cmd=cmd+' '
    return cmd,d2/180,cmd_sz1

def Gen_lng_scale():
    #train move left/right 28
    if np.random.random([1]) > 0.5:
        d2 = 'shrink'
        d3 = 0.75

    else:
        d2 = 'enlarge'
        d3=1.5
    cmd = d2
    cmd_sz1 = len(cmd)
    for ii in range(Full_len-cmd_sz1):
       cmd=cmd+' '
    return cmd,d3,cmd_sz1


def Gen_lng_thisis(num):
    d3=num
    cmd='this is '+str(num)
    cmd_sz1 = len(cmd)
    for ii in range(Full_len - cmd_sz1):
        cmd = cmd + ' '
    return cmd,d3,cmd_sz1

def Gen_lng_size():
    sz=np.random.random([1])
    if sz > 0.5:
        a0=1.2;a1=1.5
        scale=np.random.random(1)*(a1-a0)+a0
        d2 = 'big '
    else:
        a0 = 0.7;a1 = 0.85
        scale = np.random.random(1) * (a1 - a0) + a0
        d2 = 'small '
    cmd='the size is '+d2
    cmd_sz1 = len(cmd)
    for ii in range(Full_len - cmd_sz1):
        cmd = cmd + ' '
    return cmd,scale,cmd_sz1

def Gen_lng_size_not():
    sz=np.random.random([1])
    if sz > 0.5:
        a0=1.2;a1=1.5
        scale=np.random.random(1)*(a1-a0)+a0
        d2 = 'small '
    else:
        a0 = 0.7;a1 = 0.85
        scale = np.random.random(1) * (a1 - a0) + a0
        d2 = 'big '
    cmd='the size is not '+d2
    cmd_sz1 = len(cmd)
    for ii in range(Full_len - cmd_sz1):
        cmd = cmd + ' '
    return cmd,scale,cmd_sz1
def Gen_lng_giveme(num):
    d3=num
    cmd='give me a '+str(num)
    cmd_sz1 = len(cmd)
    for ii in range(Full_len - cmd_sz1):
        cmd = cmd + ' '
    return cmd,d3,cmd_sz1


TIME_STEP=30;
BATCH_SIZE=128;
tot_episodes=1000000



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
IPS_lstm_kernel=np.load('/Users/fengqi/Pycharm_py36/QF/IPS_lstm_kernel.npy')
IPS_lstm_bias=np.load('/Users/fengqi/Pycharm_py36/QF/IPS_lstm_bias.npy')
IPS_dense_kernel=np.load('/Users/fengqi/Pycharm_py36/QF/IPS_dense_kernel.npy')
IPS_dense_bias=np.load('/Users/fengqi/Pycharm_py36/QF/IPS_dense_bias.npy')


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


# IPS  speech->number

IPS_INPUT_SIZE=6;IPS_CELL_SIZE=10;IPS_OUT_SIZE=1;

IPS_x = tf.placeholder(tf.float32, [None, TIME_STEP,IPS_INPUT_SIZE])

IPS_cell = tf.contrib.rnn.BasicLSTMCell(num_units=IPS_CELL_SIZE)
init_s = IPS_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
IPS_outputs, IPS_final_s = tf.nn.dynamic_rnn(
                                        IPS_cell,                   # cell you have chosen
                                        IPS_x,                      # input
                                        initial_state=init_s,       # the initial hidden state
                                        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                                      )
IPS_outs2D = tf.reshape(IPS_outputs, [-1, IPS_CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
IPS_net_outs2D = tf.layers.dense(IPS_outs2D, IPS_OUT_SIZE,kernel_initializer=tf.constant_initializer(IPS_dense_kernel),
                                 bias_initializer=tf.constant_initializer(IPS_dense_bias),trainable=False)
IPS_outs = tf.reshape(IPS_net_outs2D, [-1, TIME_STEP, IPS_OUT_SIZE])          # reshape back to 3D



# dlPFC
dlPFC_INPUT_SIZE = 71;
dlPFC_CELL_SIZE = 128;
PFC_LearningRate = 0.001;
dlPFC_OUT_SIZE = 32 + 6;

dlPFC_x = tf.placeholder(tf.float32, [None, TIME_STEP - 1, dlPFC_INPUT_SIZE])
dlPFC_y = tf.placeholder(tf.float32, [None, TIME_STEP - 1, dlPFC_OUT_SIZE])

with tf.variable_scope('dlPFC'):
    dlPFC_cell = tf.contrib.rnn.BasicLSTMCell(num_units=dlPFC_CELL_SIZE)
    dlPFC_init_s = dlPFC_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
    dlPFC_outputs, dlPFC_final_s = tf.nn.dynamic_rnn(dlPFC_cell,  dlPFC_x,  initial_state=dlPFC_init_s,       # the initial hidden state
                                            time_major=False, )# False: (batch, time step, input); True: (time step, batch, input)
    dlPFC_outs2D = tf.reshape(dlPFC_outputs, [-1, dlPFC_CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
    dlPFC_net_outs2D = tf.layers.dense(dlPFC_outs2D, dlPFC_OUT_SIZE)
    dlPFC_outs = tf.reshape(dlPFC_net_outs2D, [-1, TIME_STEP-1, dlPFC_OUT_SIZE])          # reshape back to 3D

    dlPFC_loss = tf.losses.mean_squared_error(labels=dlPFC_y, predictions=dlPFC_outs)
        #tf.losses.mean_squared_error(labels=dlPFC_y[:,25,:], predictions=dlPFC_outs[:,25,:])  # tf.losses.mean_squared_error(labels=dlPFC_y, predictions=dlPFC_outs)+
    dlPFC_train = tf.train.AdamOptimizer(learning_rate=PFC_LearningRate).minimize(dlPFC_loss)


# Imagination network
dcd_in=tf.placeholder(tf.float32,[None,32])
# decoder
ff2 = tf.layers.dense(dcd_in, n_encoded, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wem),bias_initializer=tf.constant_initializer(bem),trainable=False)
de0 = tf.layers.dense(ff2, n_d0, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd0),bias_initializer=tf.constant_initializer(bd0),trainable=False)
de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd1),bias_initializer=tf.constant_initializer(bd1),trainable=False)
de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd2),bias_initializer=tf.constant_initializer(bd2),trainable=False)
de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wd3),bias_initializer=tf.constant_initializer(bd3),trainable=False)
decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid, tf.nn.sigmoid,kernel_initializer=tf.constant_initializer(wdd),bias_initializer=tf.constant_initializer(bdd),trainable=False)




sess = tf.Session()
sess.run(tf.global_variables_initializer())
weights_lstm_w = tf.get_default_graph().get_tensor_by_name('rnn/basic_lstm_cell/kernel:0')
weights_lstm_b = tf.get_default_graph().get_tensor_by_name('rnn/basic_lstm_cell/bias:0')
sess.run(tf.assign(weights_lstm_w, IPS_lstm_kernel))
sess.run(tf.assign(weights_lstm_b, IPS_lstm_bias))
# print(sess.run(weights_lstm_w))
# print(sess.run(weights_lstm_b))

chk_num=50000
saver = tf.train.Saver()
path = '/Users/fengqi/Pycharm_py36/QF/_Imagination' + str(chk_num) + '/'
file_='CKPT001'
saver.restore(sess, path+file_)

aaa=1


aa=tf.global_variables()
IPS_lstm_kernel=sess.run(aa[14])
np.save('PFC_lstm_kernel.npy',IPS_lstm_kernel)
IPS_lstm_bias=sess.run(aa[15])
np.save('PFC_lstm_bias.npy',IPS_lstm_bias)
IPS_dense_kernel=sess.run(aa[16])
np.save('PFC_dense_kernel.npy',IPS_dense_kernel)
IPS_dense_bias=sess.run(aa[17])
np.save('PFC_dense_bias.npy',IPS_dense_bias)
