import tensorflow as tf
import numpy as np



#read data
def readData_multi():
	path = "C:/Users/24400/Desktop/train_set_multi.tfrecords"

	sess = tf.Session()

	for serialized_example in tf.python_io.tf_record_iterator(path):

		feature = {}
		feature['img'] = tf.FixedLenFeature([], tf.string)
		i=0
		while i!=15:
			index = "label_"+str(i)
			feature[index] = tf.FixedLenFeature([], tf.string)
			i+=1
		features = tf.parse_single_example(serialized_example,features = feature)

		img = tf.decode_raw(features['img'], tf.uint8)

		img = tf.reshape(img,[96,96,1])

		print(img)

def readData_single():
	path = "C:/Users/24400/Desktop/train_set_single.tfrecords"

	sess = tf.Session()

	for serialized_example in tf.python_io.tf_record_iterator(path):

		feature = {}
		feature['img'] = tf.FixedLenFeature([], tf.string)
		feature['label'] = tf.FixedLenFeature([], tf.string)
		features = tf.parse_single_example(serialized_example,features = feature)

		img = tf.decode_raw(features['img'], tf.uint8)

		img = tf.reshape(img,[96,96,1])

		print(img)

keep_prob = None

unPooling = []

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)    
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')# step is two

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

def deconv(x,W,O):
	return tf.nn.conv2d_transpose(x,W,output_shape = O,strides=[1,2,2,1],padding = 'VALID')

def merge(convo_layer,unsampling)
	return tf.concat(values = [convo_layer,unsampling],axis = -1)


x_ = tf.placeholder(tf.float32,[None,96*96])#input
y_ = tf.placeholder(tf.float32,[None,96*96])#output


X = tf.reshape(x_,shape = [-1,96*96*1])

#first convolution 96*96 -->48*48

#---------conv1----------

w_conv = weight_variable([4,4,1,32])
b_conv = bias_variable([32])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

# ---------conv2-----------
w_conv = weight_variable([4,4,32,32])
b_conv = bias_variable([32])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

unPooling.append(X)

#---------maxpool--------

img_pool = max_pooling(img_conv)

X = img_pool

X = tf.nn.dropout(X,keep_prob = keep_prob)

#second convolution 48*48 --> 24*24

#---------conv1----------

w_conv = weight_variable([4,4,32,64])
b_conv = bias_variable([64])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

# ---------conv2-----------
w_conv = weight_variable([4,4,64,64])
b_conv = bias_variable([64])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

unPooling.append(X)

#---------maxpool--------

img_pool = max_pooling(img_conv)

X = img_pool

X = tf.nn.dropout(X,keep_prob = keep_prob)


#third convolution 24*24 -->12*12 

#---------conv1----------

w_conv = weight_variable([4,4,64,128])
b_conv = bias_variable([128])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

# ---------conv2-----------
w_conv = weight_variable([4,4,128,128])
b_conv = bias_variable([128])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

unPooling.append(X)

#---------maxpool--------

img_pool = max_pooling(img_conv)

X = img_pool

X = tf.nn.dropout(X,keep_prob = keep_prob)


#bottom convolution 

#---------conv1----------

w_conv = weight_variable([3,3,128,256])
b_conv = bias_variable([256])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

# ---------conv2-----------
w_conv = weight_variable([3,3,256,256])
b_conv = bias_variable([256])

img_conv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_conv

#---------usample--------
w_conv = weight_variable([2,2,128,256])
b_conv = bias_variable([256])

img_deconv = deconv(img_conv,w_conv,[1,24,24,128])

X = img_deconv

img_deconv = tf.nn.relu(tf.nn.bias_add(X,b_conv))

X = img_deconv

X = tf.nn.dropout(X,keep_prob = keep_prob)

#---------upsample1----------------


X = merge(unPooling[2],X)

w_conv = weight_variable([4,4,128,64])
b_conv = bias_variable([64])

img_deconv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([4,4,64,64])
b_conv = bias_variable([64])

img_deconv = tf.nn relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([2,2,64,128])
b_conv = bias_variable([64])

img_deconv = deconv(img_deconv,w_conv,[1,48,48,64])

img_deconv = tf.nn.relu(tf.nn.bias_add(X,b_conv))

X = img_deconv

X = tf.nn.dropout(X,keep_prob = keep_prob)

# ----------unsample2-------------------

X = merge(unPooling[1],X)

w_conv = weight_variable([4,4,64,32])
b_conv = bias_variable([32])

img_deconv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([4,4,32,32])
b_conv = bias_variable([32])

img_deconv = tf.nn relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([2,2,32,64])
b_conv = bias_variable([32])

img_deconv = deconv(img_deconv,w_conv,[1,96,96,32])

img_deconv = tf.nn.relu(tf.nn.bias_add(X,b_conv))

X = img_deconv

X = tf.nn.dropout(X,keep_prob = keep_prob)


#-----------------final layer--------------
X = merge(unPooling[1],X)

w_conv = weight_variable([4,4,64,32])
b_conv = bias_variable([32])

img_deconv = tf.nn.relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([4,4,32,32])
b_conv = bias_variable([32])

img_deconv = tf.nn relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

w_conv = weight_variable([1,1,32,2])
b_conv = bias_variable([2])

img_deconv = tf.nn relu(conv2d(X,w_conv)+b_conv)

X = img_deconv

#softmax loss

def loss():
	return tf.nn.sparse_softmax_cross_entropy_with_logits(labels = ,logits = )

def loss_mean():
	return tf.reduce_mean(loss())

def loss_all() = tf.add_n(inputs = tf.get_collection(key= 'loss'))






