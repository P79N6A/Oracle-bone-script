import os,shutil,stat
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import Augmentor
import cv2
import random

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


train_set_writer = tf.python_io.TFRecordWriter("C:/Users/24400/Desktop/train_set_cnn.tfrecords")

def writeToSet():
	path1 = "C:/Users/24400/Desktop/oracle-jpg"
	path2 = "C:/Users/24400/Desktop/jin-jpg"

	temp = os.listdir(path1)
	image = []
	for img in temp:
		image.append(path1+"/"+img)

	temp = os.listdir(path2)
	label = []
	for lab in temp:
		label.append(path2+"/"+lab)

	index = 0
	while index!=420:
		trueList = []
		falseList = []
		imageList = []
		tempImg = os.listdir(image[index])
		tempLabel = os.listdir(label[index])
		trueNum = len(tempLabel)
		for element in tempImg:
			imagePath = image[index]+"/"+element
			imageList.append(imagePath)
			num = random.randint(0,trueNum-1)
			labelPath = label[index]+"/"+tempLabel[num]
			trueList.append(labelPath)
			falseNum1 = 0
			falseNum2 = 0
			if index>210:
				falseNum1 = random.randint(0,index-1)
				tempFalse = os.listdir(label[falseNum1])
				tempNum = len(tempFalse)
				falseNum2 = random.randint(0,tempNum-1)
				falsePath = label[falseNum1]+"/"+tempFalse[falseNum2]
				falseList.append(falsePath)
			else:
				falseNum1 = random.randint(index+1,419)
				tempFalse = os.listdir(label[falseNum1])
				tempNum = len(tempFalse)
				falseNum2 = random.randint(0,tempNum-1)
				falsePath = label[falseNum1]+"/"+tempFalse[falseNum2]
				falseList.append(falsePath)

		print(imageList,trueList,falseList)
		
		value = len(imageList)
		i = 0
		while i!=value:
			image1 = imageList[i]
			image1 = cv2.imdecode(np.fromfile(image1,dtype=np.uint8),-1)
			image1 = cv2.resize(src = image1,dsize=(96,96))
			image1 = convertToBinary(image1)

			image2 = trueList[i]
			image2 = cv2.imdecode(np.fromfile(image2,dtype=np.uint8),-1)
			image2 = cv2.resize(src = image2,dsize=(96,96))
			image2 = convertToBinary(image2)

			image3 = cv2.imdecode(np.fromfile(image3,dtype=np.uint8),-1)
			image3 = cv2.resize(src = image3,dsize=(96,96))
			image3 = convertToBinary(image3)

			feature = {}
			feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image1.tobytes()]))
			feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
			feature['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[1]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())

			feature = {}
			feature['img1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image2.tobytes()]))
			feature['img2'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image3.tobytes()]))
			feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			train_set_writer.write(example.SerializeToString())

			i+=1
		
		index+=1
		if index%10==0:
			print(index)
	print("Done")

writeToSet()
train_set_writer.close()
