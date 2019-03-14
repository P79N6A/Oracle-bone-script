import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

def augment():
	
	p = Augmentor.Pipeline(
		source_directory="C:/Users/24400/Desktop/temp",
		output_directory="C:/Users/24400/Desktop/label"
	)
	p.rotate(probability=0.5, max_left_rotation=2, max_right_rotation=2)
	p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
	p.skew(probability=0.5)
	p.random_distortion(probability=0.5, grid_width=96, grid_height=96, magnitude=1)
	p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.sample(n=1000)

train_set_writer = tf.python_io.TFRecordWriter("C:/Users/24400/Desktop/train_set_Unet.tfrecords") 

def binaryToImg(bin):
	axis = []
	for x in bin:
		element = []
		for y in x:
			temp = []
			if y == 1:
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


def writeToSet2(image_path,label_path):
	imgPath = os.listdir(image_path)

	tempLabel = os.listdir(label_path)

	train_label = []
	train = []
	for label in tempLabel:
		tempPath = label_path+"/"+label
		label_img = cv2.imdecode(np.fromfile(tempPath,dtype=np.uint8),-1)
		label_img = cv2.resize(src = label_img,dsize=(96,96))
		label_img = convertToBinary(label_img)
		train_label.append(label_img)

	for img in imgPath:
		tempImg = image_path+"/"+img
		train_img = cv2.imdecode(np.fromfile(tempImg,dtype=np.uint8),-1)
		train_img = cv2.resize(src = train_img,dsize=(96,96))
		train_img = convertToBinary(train_img)
		train.append(train_img)

	num = 0
	for img in train:
		newLabel = []
		if len(train_label)>3:
			newLabel = random.sample(train_label,3)
		else:
			newLabel = train_label
		for label in newLabel:
			feature = {}
			feature['img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
			feature['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())
			num+=1
	return num

	
path1 = "C:/Users/24400/Desktop/oracle-jpg"
path2 = "C:/Users/24400/Desktop/jin-jpg"

chars = os.listdir(path1)
 
index = 0

for char in chars:
	tempPath1 = path1+"/"+char
	tempPath2 = path2+"/"+char
	num = writeToSet2(tempPath1,tempPath2)
	index+=1
	print(index,num)

train_set_writer.close()
print("done")

