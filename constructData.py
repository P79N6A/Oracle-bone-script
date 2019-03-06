import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

def convert():
	sourcePath = "C:/Users/24400/Desktop/jin/安"
	imgPath = os.listdir(sourcePath)
	for img in imgPath:
		path = sourcePath+"/"+img
		print(path)
		im = Image.open(path)
		tempImg = img.split(".")
		im.save("C:/Users/24400/Desktop/label/安"+"/"+tempImg[0]+".png")


def augment():
	
	p = Augmentor.Pipeline(
		source_directory="C:/Users/24400/Desktop/train/安",
		output_directory="C:/Users/24400/Desktop/train/安1"
	)
	p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
	p.skew(probability=0.2)
	p.random_distortion(probability=0.2, grid_width=96, grid_height=96, magnitude=1)
	p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.sample(n=100)

train_set_writer = tf.python_io.TFRecordWriter("C:/Users/24400/Desktop/train_set_single.tfrecords") 

def writeToSet(image_path,label_path):

	imgPath = os.listdir(image_path)

	tempLabel = os.listdir(label_path)
	train_label = []
	for label in tempLabel:
		tempPath = label_path+"/"+label
		#label_img = cv2.imread(tempPath)
		label_img = cv2.imdecode(np.fromfile(tempPath,dtype=np.uint8),-1)
		label_img = cv2.resize(src = label_img,dsize=(96,96))
		train_label.append(label_img)
	for img in imgPath:
		tempImg = image_path+"/"+img
		#train_img = cv2.imread(tempImg)
		train_img = cv2.imdecode(np.fromfile(tempImg,dtype=np.uint8),-1)
		train_img = cv2.resize(src = train_img,dsize=(96,96))
		feature = {}
		feature['img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_img.tobytes()]))
		i=0
		for element in train_label:
			index = "label_"+str(i)
			feature[index] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[element.tobytes()]))
			i+=1

		example = tf.train.Example(features = tf.train.Features(feature = feature))
		train_set_writer.write(example.SerializeToString())


def writeToSet2(image_path,label_path):
	imgPath = os.listdir(image_path)

	tempLabel = os.listdir(label_path)

	train_label = []
	train = []
	for label in tempLabel:
		tempPath = label_path+"/"+label
		#label_img = cv2.imread(tempPath)
		label_img = cv2.imdecode(np.fromfile(tempPath,dtype=np.uint8),-1)
		label_img = cv2.resize(src = label_img,dsize=(96,96))
		train_label.append(label_img)

	for img in imgPath:
		tempImg = image_path+"/"+img
		#train_img = cv2.imread(tempImg)
		train_img = cv2.imdecode(np.fromfile(tempImg,dtype=np.uint8),-1)
		train_img = cv2.resize(src = train_img,dsize=(96,96))
		train.append(train_img)

	for img in train:
		newLabel = []
		if len(train_label)>3:
			newLabel = random.sample(train_label,3)
		else:
			newLabel = train_label
			for label in train_label:
				feature = {}
				feature['img'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
				feature['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
				example = tf.train.Example(features = tf.train.Features(feature = feature))
				train_set_writer.write(example.SerializeToString())


	
path1 = "C:/Users/24400/Desktop/oracle"
path2 = "C:/Users/24400/Desktop/jin"

chars = os.listdir(path1)
 
index = 0
for char in chars:
	tempPath1 = path1+"/"+char
	tempPath2 = path2+"/"+char
	writeToSet2(tempPath1,tempPath2)
	index+=1
	print(index)

train_set_writer.close()
print("done")