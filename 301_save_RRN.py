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

#mv x=deltaX
def affine_transform_delta_x(b_x,deltaX):
    xx = deltaX  # (-0.5,0.5)
    yy = 0
    sx = 1 # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_delta_y(b_x,deltaY):
    xx = 0 # (-0.5,0.5)
    yy = deltaY
    sx = 1 # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x


# shrink to 0.60
def affine_transform_60(b_x):

    xx = np.random.uniform(low=0, high=0, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=0, high=0, size=1)
    sx = np.random.uniform(low=0.60, high=0.60, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=0.60, high=0.60, size=1)
    rr = np.random.uniform(low=0, high=0, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

# mv x
def affine_transform_rand_x(b_x):
    xx = np.random.uniform(low=-0.2, high=0.2, size=1) # (-0.5,0.5)
    yy = 0
    sx = 1  # (.5,1.2)
    sy = 1
    rr = 0 # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

# mv y
def affine_transform_rand_y(b_x):
    xx = 0  # (-0.5,0.5)
    yy = np.random.uniform(low=-0.2, high=0.2, size=1)
    sx = 1  # (.5,1.2)
    sy = 1
    rr = 0 # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

# rtt
def affine_transform_rand_r(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = 1  # (.5,1.2)
    sy = 1
    rr = np.random.uniform(low=-0.2, high=0.2, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_rand_s(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    s=np.random.uniform(low=0.5, high=0.7, size=1)  # (-0.5,0.5)
    sx = s  # (.5,1.2)
    sy = s
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
#scale x=deltaS
def affine_transform_delta_s(b_x,deltaS):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = deltaS # (.5,1.2)
    sy = deltaS
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_rand_sx(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx =np.random.uniform(low=0.5, high=0.7, size=1)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
def affine_transform_rand_sx_full(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx =np.random.uniform(low=0.3, high=1, size=1)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x


def affine_transform_delta_sx(b_x,deltaS ):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = deltaS # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_delta_r(b_x,deltaR):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = 1 # (.5,1.2)
    sy = 1
    rr = deltaR  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x



def square(b_x):
    b_x=b_x*0+1
    xx = 0 # (-0.5,0.5)
    yy = 0
    sx = 1 # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x




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
    return b_x, np.concatenate((xx, yy, rr), axis=0)


def affine_transform_rand_noScale(b_x):
    xx = np.random.uniform(low=-0.2, high=.2, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-.2, high=.2, size=1)
    sx = np.random.uniform(low=0.8, high=1.2, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=0.8, high=1.2, size=1)
    rr = np.random.uniform(low=-1, high=1, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x



def cal_w_stats(we, wep):
    wed = we - wep;
    wed_max = np.amax(wed);
    wed_min = np.amin(wed);
    wed_avg = np.average(wed);
    wed_abs_avg = np.average(np.abs(wed));
    we_std = np.std(we);
    we_avg = np.average(we)
    we_absavg = np.average(np.abs(we))
    return np.concatenate((wed_max.reshape([1]), wed_min.reshape([1]), wed_avg.reshape([1]), wed_abs_avg.reshape([1]),
                           we_std.reshape([1]), we_avg.reshape([1]),we_absavg.reshape([1])))

def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = plt.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)




BATCH_SIZE=128
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

print(n_encoded)
type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(XX) + '-' + str(
    SX1) + '-' + str(SX2) + '-' + str(RX) + '-'

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28 * 28])  # value in the range of (0, 1)
ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
ph_switch = tf.placeholder(tf.float32, [1])
ph_lr = tf.placeholder(tf.float32, [])
ph_dis_e = tf.placeholder(tf.float32, [None, n_encoded])
# encoder


en0 = tf.layers.dense(tf_x, n_l0, tf.nn.sigmoid)
en1 = tf.layers.dense(en0, n_l1, tf.nn.sigmoid)
en2 = tf.layers.dense(en1, n_l2, tf.nn.sigmoid)
en3 = tf.layers.dense(en2, n_l3, tf.nn.sigmoid)
ff_encoded = tf.layers.dense(en3, n_encoded, tf.nn.sigmoid)
enc = ff_encoded * ph_switch + ph_encoded * (1 - ph_switch)
encoded = tf.multiply(enc, ph_dis_e)
# decoder
de0 = tf.layers.dense(encoded, n_d0, tf.nn.sigmoid)
de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid)
de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid)
de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid)
decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(ph_lr).minimize(loss)

weights_en0 = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')
weights_en1 = tf.get_default_graph().get_tensor_by_name(os.path.split(en1.name)[0] + '/kernel:0')
weights_en2 = tf.get_default_graph().get_tensor_by_name(os.path.split(en2.name)[0] + '/kernel:0')
weights_en3 = tf.get_default_graph().get_tensor_by_name(os.path.split(en3.name)[0] + '/kernel:0')
weights_mid = tf.get_default_graph().get_tensor_by_name(os.path.split(ff_encoded.name)[0] + '/kernel:0')
weights_de0 = tf.get_default_graph().get_tensor_by_name(os.path.split(de0.name)[0] + '/kernel:0')
weights_de1 = tf.get_default_graph().get_tensor_by_name(os.path.split(de1.name)[0] + '/kernel:0')
weights_de2 = tf.get_default_graph().get_tensor_by_name(os.path.split(de2.name)[0] + '/kernel:0')
weights_de3 = tf.get_default_graph().get_tensor_by_name(os.path.split(de3.name)[0] + '/kernel:0')
weights_ddr = tf.get_default_graph().get_tensor_by_name(os.path.split(decoded.name)[0] + '/kernel:0')


bias_en0 = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/bias:0')
bias_en1 = tf.get_default_graph().get_tensor_by_name(os.path.split(en1.name)[0] + '/bias:0')
bias_en2 = tf.get_default_graph().get_tensor_by_name(os.path.split(en2.name)[0] + '/bias:0')
bias_en3 = tf.get_default_graph().get_tensor_by_name(os.path.split(en3.name)[0] + '/bias:0')
bias_mid = tf.get_default_graph().get_tensor_by_name(os.path.split(ff_encoded.name)[0] + '/bias:0')
bias_de0 = tf.get_default_graph().get_tensor_by_name(os.path.split(de0.name)[0] + '/bias:0')
bias_de1 = tf.get_default_graph().get_tensor_by_name(os.path.split(de1.name)[0] + '/bias:0')
bias_de2 = tf.get_default_graph().get_tensor_by_name(os.path.split(de2.name)[0] + '/bias:0')
bias_de3 = tf.get_default_graph().get_tensor_by_name(os.path.split(de3.name)[0] + '/bias:0')
bias_ddr = tf.get_default_graph().get_tensor_by_name(os.path.split(decoded.name)[0] + '/bias:0')





sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
type ='/Users/fengqi/Pycharm_py36/QF/4900000/' + '_n_32-batch128-lr0.001-0.2-0.5-0.7-0.2-nx0'
saver.restore(sess, type)
BATCH_SIZE=128
mnist = input_data.read_data_sets('./mnist', one_hot=False)

tf.set_random_seed(1)
LearningRate = 0.001;ph_lr_ = np.ones(shape=[]) * LearningRate


b_x = mnist.test.images[:BATCH_SIZE]
b_y = mnist.test.labels[:BATCH_SIZE]
test_x=np.zeros([BATCH_SIZE,784])


VIEW_SIZE=128
ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])

view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3,wdd,be0, be1, be2, be3, bem, bd0, bd1, bd2, bd3, bdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr,
     bias_en0, bias_en1, bias_en2, bias_en3, bias_mid,
     bias_de0, bias_de1, bias_de2, bias_de3, bias_ddr,
     en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: b_x,
     ph_encoded: ph_encoded_,
     ph_switch: ph_switch_,
     ph_dis_e: ph_dis_e_})

np.save('we0.npy',we0)
np.save('we1.npy',we1)
np.save('we2.npy',we2)
np.save('we3.npy',we3)
np.save('wem.npy',wem)
np.save('wd0.npy',wd0)
np.save('wd1.npy',wd1)
np.save('wd2.npy',wd2)
np.save('wd3.npy',wd3)
np.save('wdd.npy',wdd)

np.save('be0.npy',be0)
np.save('be1.npy',be1)
np.save('be2.npy',be2)
np.save('be3.npy',be3)
np.save('bem.npy',bem)
np.save('bd0.npy',bd0)
np.save('bd1.npy',bd1)
np.save('bd2.npy',bd2)
np.save('bd3.npy',bd3)
np.save('bdd.npy',bdd)




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