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



'''
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
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')

def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

'''
