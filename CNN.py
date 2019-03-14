import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as Ige


def convertToBinary(img):
	axis = []
	for x in img:
		element = []	
		for y in x:
			if y[0]<150:
				element.append(1)
			else:
				element.append(0)
		element = np.array(element,dtype = 'uint8')
		axis.append(element)
	axis = np.array(axis)
	return axis

def binaryToImg(bin):
	axis = []
	for xAxis in bin:
		element = []
		for yAxis in xAxis:
			temp = []
			if yAxis == 1:
				temp.append(0)
				temp.append(0)
				temp.append(0)
			else:
				temp.append(255)
				temp.append(255)
				temp.append(255)
			temp = np.array(temp,dtype = 'uint8')
			element.append(temp)
		element = np.array(element)
		axis.append(element)

	axis = np.array(axis)
	return axis


def readData_single():
	path = "C:/Users/24400/Desktop/train_set_cnn.tfrecords"

	filename_queue = tf.train.string_input_producer([path],num_epochs = 1,shuffle = True)

	reader = tf.TFRecordReader()

	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example,features = {
			'img1': tf.FixedLenFeature([], tf.string),
			'img2': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int32)
		})

	image1 = tf.decode_raw(features['img1'],tf.uint8)

	image1 = tf.reshape(image,[96,96,1])

	image2 = tf.decode_raw(features['img2'],tf.uint8)

	image2 = tf.reshape(image,[96,96,1])

	label = tf.cast(features['label'], tf.int32)

	return image1,image2,label

class CNN:
	def __init__(self):
		self.input_image1 = tf.placeholder(dtype = tf.float32,shape = [1,96,96,1])

		self.input_image2 = tf.placeholder(dtype = tf.float32,shape = [1,96,96,1])

		self.label = tf.placeholder(dtype = tf.float32,shape = [None,2])

		
	def weight_variable(self,shape):
	    initial = tf.truncated_normal(shape,stddev=0.1)
	    tf.add_to_collection(name = 'loss',value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))   
	    return tf.Variable(initial)

	def bias_variable(self,shape):
	    initial = tf.random_normal(shape=shape,dtype = tf.float32)
	    return tf.Variable(initial_value = initial)


	def conv2d(self,x,W):
	    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')

	def max_pooling(self,x):
	    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

	def setup_network(self):
		