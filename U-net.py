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
	path = "C:/Users/24400/Desktop/train_set_Unet.tfrecords"

	filename_queue = tf.train.string_input_producer([path],num_epochs = 1,shuffle = True)

	reader = tf.TFRecordReader()

	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(serialized_example,features = {
			'img': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features['img'],tf.uint8)

	image = tf.reshape(image,[96,96,1])

	label = tf.decode_raw(features['label'],tf.uint8)

	label = tf.reshape(label,[96,96])

	return image,label
	

class Unet:

	def __init__(self):
	
		self.keep_prob = tf.placeholder(dtype=tf.float32)

		self.lamb = tf.placeholder(dtype = tf.float32)

		self.unPooling = []

		self.input_image = tf.placeholder(dtype = tf.float32,shape = [1,96,96,1])

		self.input_label = tf.placeholder(dtype = tf.int32,shape = [1,96,96])

		self.prediction = None

		self.correct_prediction = None

		self.accurancy = None

		self.loss = None

		self.loss_mean = None

		self.loss_all = None

		self.train_step = None



	def weight_variable(self,shape):
	    initial = tf.truncated_normal(shape,stddev=0.015)
	    tf.add_to_collection(name = 'loss',value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))   
	    return tf.Variable(initial)

	def bias_variable(self,shape):
	    initial = tf.random_normal(shape=shape,dtype = tf.float32)
	    return tf.Variable(initial_value = initial)


	def conv2d(self,x,W):
	    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')

	def max_pooling(self,x):
	    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

	def deconv(self,x,W,O):
		return tf.nn.conv2d_transpose(value = x,filter = W,output_shape = O,strides=[1,2,2,1],padding = 'VALID')

	def merge_img(self,convo_layer,unsampling):
		return tf.concat(values = [convo_layer,unsampling],axis = -1)

	def setup_network(self):


		#first convolution 96*96 -->48*48

		
		with tf.name_scope('first_convolution'):

			#---------conv1----------

			w_conv = self.weight_variable([4,4,1,32])
			b_conv = self.bias_variable([32])

			img_conv = tf.nn.relu(self.conv2d(self.input_image,w_conv)+b_conv)

			X = img_conv

			# ---------conv2-----------
			w_conv = self.weight_variable([4,4,32,32])
			b_conv = self.bias_variable([32])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			self.unPooling.append(X)

			#---------maxpool--------

			img_pool = self.max_pooling(img_conv)

			X = img_pool


		#second convolution 48*48 --> 24*24

		with tf.name_scope('second_convolution'):

			#---------conv1----------

			w_conv = self.weight_variable([4,4,32,64])
			b_conv = self.bias_variable([64])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			# ---------conv2-----------
			w_conv = self.weight_variable([4,4,64,64])
			b_conv = self.bias_variable([64])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			self.unPooling.append(X)

			#---------maxpool--------

			img_pool = self.max_pooling(img_conv)

			X = img_pool



		#third convolution 24*24 -->12*12 

		with tf.name_scope('third_convolution'):

			#---------conv1----------

			w_conv = self.weight_variable([4,4,64,128])
			b_conv = self.bias_variable([128])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			# ---------conv2-----------
			w_conv = self.weight_variable([4,4,128,128])
			b_conv = self.bias_variable([128])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			self.unPooling.append(X)

			#---------maxpool--------

			img_pool = self.max_pooling(img_conv)

			X = img_pool

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)


		#bottom convolution 

		
		with tf.name_scope('bottom_convolution'):

			#---------conv1----------

			w_conv = self.weight_variable([3,3,128,256])
			b_conv = self.bias_variable([256])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			# ---------conv2-----------
			w_conv = self.weight_variable([3,3,256,256])
			b_conv = self.bias_variable([256])

			img_conv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_conv

			#---------usample--------
			w_conv = self.weight_variable([2,2,128,256])
			b_conv = self.bias_variable([128])


			img_deconv = tf.nn.relu(self.deconv(img_conv,w_conv,[1,24,24,128])+b_conv)
			X = img_deconv



		with tf.name_scope('first_deconvolution'):


			#first deconvolution

			X = self.merge_img(self.unPooling[2],X)

			w_conv = self.weight_variable([4,4,256,128])
			b_conv = self.bias_variable([128])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([4,4,128,128])
			b_conv = self.bias_variable([128])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([2,2,64,128])
			b_conv = self.bias_variable([64])


			img_deconv = tf.nn.relu(self.deconv(img_deconv,w_conv,[1,48,48,64])+b_conv)


			X = img_deconv


		with tf.name_scope('second_deconvolution'):

			# second deconvolution

			X = self.merge_img(self.unPooling[1],X)

			w_conv = self.weight_variable([4,4,128,64])
			b_conv = self.bias_variable([64])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([4,4,64,64])
			b_conv = self.bias_variable([64])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([2,2,32,64])
			b_conv = self.bias_variable([32])


			img_deconv = tf.nn.relu(self.deconv(img_deconv,w_conv,[1,96,96,32])+b_conv)


			X = img_deconv

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)


		with tf.name_scope('final_layer'):

			#final layer

			X = self.merge_img(self.unPooling[0],X)

			w_conv = self.weight_variable([4,4,64,32])
			b_conv = self.bias_variable([32])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([4,4,32,32])
			b_conv = self.bias_variable([32])

			img_deconv = tf.nn.relu(self.conv2d(X,w_conv)+b_conv)

			X = img_deconv

			w_conv = self.weight_variable([1,1,32,2])
			b_conv = self.bias_variable([2])

			img_deconv = tf.nn.conv2d(input = X,filter = w_conv,strides = [1,1,1,1],padding = 'VALID')

			self.prediction = tf.nn.bias_add(img_deconv,b_conv)


		#softmax loss

		with tf.name_scope('softmax'):

			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.input_label,logits = self.prediction, name = 'loss')

			self.loss_mean = tf.reduce_mean(self.loss)

			tf.add_to_collection(name = 'loss',value=self.loss_mean)

			self.loss_all = tf.add_n(inputs = tf.get_collection(key= 'loss'))



		with tf.name_scope('accurancy'):

			self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)

			self.correct_prediction = tf.cast(self.correct_prediction,tf.float32)

			self.accurancy = tf.reduce_mean(self.correct_prediction)

		with tf.name_scope('gradient_descent'):

			self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_all)



	def train(self):

		ckpt_path = "C:/Users/24400/Desktop/ckpt/model.ckpt"
		
		tf.summary.scalar("loss", self.loss_mean)
		
		tf.summary.scalar('accuracy', self.accurancy)
		
		merged_summary = tf.summary.merge_all()


		model_dir = "C:/Users/24400/Desktop/data/model"

		tb_dir = "C:/Users/24400/Desktop/data/logs"

		all_parameters_saver = tf.train.Saver()


		with tf.Session() as sess:

			image,label = readData_single()

			image_batch,label_batch = tf.train.shuffle_batch([image,label],batch_size = 1,num_threads = 4,capacity = 1012,min_after_dequeue = 1000)


			sess.run(tf.global_variables_initializer())
			
			sess.run(tf.local_variables_initializer())

			summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)
			
			tf.summary.FileWriter(model_dir, sess.graph)
			
			coord = tf.train.Coordinator()
			
			threads = tf.train.start_queue_runners(coord = coord)

			try:

				epoch = 1

				while not coord.should_stop():


					example,label = sess.run([image_batch,label_batch])


					lo,acc,summary = sess.run([self.loss_mean,self.accurancy,merged_summary],feed_dict = {
							self.input_image:example,self.input_label:label,self.keep_prob:0.7,self.lamb:0.004
						})

					summary_writer.add_summary(summary, epoch)

					sess.run([self.train_step],feed_dict={
							self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
							self.lamb: 0.004
						})

					epoch+=1


					if epoch%10 == 0:
						print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))


			except tf.errors.OutOfRangeError:
				print('Done training -- epoch limit reached')	


			finally:
				all_parameters_saver.save(sess = sess,save_path = ckpt_path)
				coord.request_stop()

			coord.join(threads)

			print("done training")

	def estimate(self):
		imgPath = "C:/Users/24400/Desktop/J07022.jpg"

		img = cv2.imdecode(np.fromfile(imgPath,dtype=np.uint8),-1)
		img = cv2.resize(src = img,dsize=(96,96))
		img = convertToBinary(img)

		data = img

		data = np.reshape(a=data, newshape=(1, 96, 96,1))

		ckpt_path = "C:/Users/24400/Desktop/ckpt/model.ckpt"

		all_parameters_saver = tf.train.Saver()
		with tf.Session() as sess:  
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
			predict_image = sess.run(
							tf.argmax(input=self.prediction, axis=3), 
							feed_dict={
								self.input_image: data,
								self.keep_prob: 1.0, self.lamb: 0.004
							}
						)
			
			predict_image = predict_image[0]
			
			predict_image = binaryToImg(predict_image)
			predict_image = Ige.fromarray(predict_image,'RGB')
			predict_image.save('predict_image.jpg')
			predict_image.show() 
			
			
		print('Done prediction')

def main():
	unet = Unet()
	unet.setup_network()
	#unet.train()
	unet.estimate()

main()
