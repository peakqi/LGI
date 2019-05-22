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
    d1 = np.random.randint(28)
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
tot_episodes=1000



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
chk_num=5000
saver = tf.train.Saver()
path = '/Users/fengqi/Pycharm_py36/QF/ThisMoveSizeNotGive' + str(chk_num) + '/'
file_='CKPT001'
saver.restore(sess, path+file_)

spc_bi=char_to_bi(' ')
# f,a=plt.subplots(2,TIME_STEP-1)
b_x_1=np.zeros([BATCH_SIZE,784])
b_x_2=np.zeros([BATCH_SIZE,784])
cmd_bi_=np.zeros([BATCH_SIZE,TIME_STEP,6])
del_x=np.zeros([BATCH_SIZE])
en3_1=np.zeros([BATCH_SIZE,n_l3])
ff1_1 = np.zeros([BATCH_SIZE,n_encoded])
en3_2=np.zeros([BATCH_SIZE,n_l3])
ff1_2=np.zeros([BATCH_SIZE,n_encoded])
cmd_sz=np.zeros([BATCH_SIZE])

cost_his=np.zeros([tot_episodes])
ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])
ph_encoded_1 = np.zeros(shape=[1, n_encoded])
ph_switch_1 = np.ones(shape=[1])
ph_dis_e_1 = np.ones(shape=[1, n_encoded])
img_orig=np.ones([BATCH_SIZE,TIME_STEP-1,784])
img_pred=np.ones([BATCH_SIZE,TIME_STEP-1,784])
aud_=np.zeros([BATCH_SIZE,TIME_STEP,IPS_INPUT_SIZE])
en3_=np.zeros([BATCH_SIZE,TIME_STEP,n_l3])
ff1_=np.zeros([BATCH_SIZE,TIME_STEP,n_encoded])
whole_=np.zeros([BATCH_SIZE,TIME_STEP,n_encoded+n_l3+1+IPS_INPUT_SIZE])
LearningRate = 0.0001;ph_lr_ = np.ones(shape=[]) * LearningRate
mnist = input_data.read_data_sets('./mnist', one_hot=False)
LR=np.zeros([BATCH_SIZE])
for ep in range (20):#episode
    # prep inp [en3_32,ATL32,aud6,num1] total 71
    rand_0=4
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x=b_x*0
    # Prepare SOUND
    for ii in range(BATCH_SIZE):
        if rand_0 == 0:
            cmd, del_x[ii], LR[ii], cmd_sz[ii] = Gen_lng_move()
        elif rand_0 == 1:
            cmd, del_x[ii], cmd_sz[ii] = Gen_lng_thisis(b_y[ii])
        elif rand_0 == 2:
            cmd, del_x[ii], cmd_sz[ii] = Gen_lng_size()
        elif rand_0==3:
            cmd, del_x[ii], cmd_sz[ii] = Gen_lng_size_not()
        elif rand_0==4:
            cmd, del_x[ii], cmd_sz[ii] = Gen_lng_giveme(b_y[ii])
        aud_[ii, :, :] = sentence_to_vec(cmd)

    # Prep Image
    for ii in range(BATCH_SIZE):
        b_x_1[ii], para = affine_transform1(np.reshape(b_x[ii], [1, 784]))
        if rand_0 == 0:
            b_x_2[ii] = affine_transform_x(b_x_1[ii], del_x[ii])
        elif rand_0 == 1:
            b_x_2[ii] = b_x_1[ii]
        elif rand_0 == 2:
            b_x_1[ii] = affine_transform_s(np.reshape(b_x_1[ii], [1, 784]), del_x[ii])
            b_x_2[ii] = b_x_1[ii]
        elif rand_0==3:
            b_x_1[ii] = affine_transform_s(np.reshape(b_x_1[ii], [1, 784]), del_x[ii])
            b_x_2[ii] = b_x_1[ii]
        elif rand_0==4:
            b_x_1[ii] = affine_transform60(np.reshape(b_x[ii], [1, 784]))
            b_x_2[ii]=b_x_1[ii]
            b_x_1[ii]=b_x_1[ii] *0

        en3_1[ii], ff1_1[ii] = sess.run([en3, ff1], {tf_x: np.reshape(b_x_1[ii], [1, 784]), ph_encoded: ph_encoded_1,
                                                     ph_switch: ph_switch_1, ph_dis_e: ph_dis_e_1})
        en3_2[ii], ff1_2[ii] = sess.run([en3, ff1], {tf_x: np.reshape(b_x_2[ii], [1, 784]), ph_encoded: ph_encoded_1,
                                                     ph_switch: ph_switch_1, ph_dis_e: ph_dis_e_1})

    ips_out_ = sess.run(IPS_outs, {IPS_x: aud_})  # ips_out_ (128, 30, 1)
    for jj in range(BATCH_SIZE):
        for ii in range(TIME_STEP-1):
            img_orig[jj, ii] = b_x_1[jj]
            if rand_0 == 0 or rand_0==4:
                if ii <= cmd_sz[jj]:
                    whole_[jj, ii, 6:38] = en3_1[jj]
                    whole_[jj, ii, 38:70] = ff1_1[jj]
                else:
                    whole_[jj, ii, 6:38] = en3_2[jj]
                    whole_[jj, ii, 38:70] = ff1_2[jj]

            else:
                whole_[jj, ii, 6:38] = en3_1[jj]
                whole_[jj, ii, 38:70] = ff1_1[jj]

    whole_[:, :, 0:6] = aud_
    whole_[:, :, 70:71] = ips_out_

    dlPFC_outs_ = sess.run(dlPFC_outs, {dlPFC_x: whole_[:, 0:29, :], })
    f, a = plt.subplots(2, 29)
    for ii in range(29):
        img_pred[BATCH_SIZE - 1, ii] = sess.run(decoded,
                                                {dcd_in: np.reshape(dlPFC_outs_[BATCH_SIZE - 1, ii, 6:38], [1, 32])})
        aud_pred = aud_code_to_txt(np.round(dlPFC_outs_[BATCH_SIZE - 1, ii, 0:6]))
        a[0][ii].imshow(np.reshape(img_orig[BATCH_SIZE - 1, ii], [28, 28]))
        a[1][ii].imshow(np.reshape(img_pred[BATCH_SIZE - 1, ii], [28, 28]))
        a[0][ii].set_title(aud_code_to_txt(aud_[BATCH_SIZE - 1, ii]))
        a[1][ii].set_title(aud_pred)
        a[0][ii].set_xticks(());
        a[0][ii].set_yticks(());
        a[1][ii].set_xticks(());
        a[1][ii].set_yticks(());
    rand_1 = np.random.random(1)
    file_nm = '/Users/fengqi/Pycharm_py36/QF/ThisMovieSizeNotGive' + str(chk_num) + str(rand_1) + '.jpeg'
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(file_nm, dpi=100)
    plt.close('all')
