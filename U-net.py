import tensorflow as tf
import numpy as np


def readData_single(file_queue):

	reader = tf.TFRecordReader()

	name,serialized_example = reader.read(file_queue)

	features = tf.parse_single_example(serialized_example,features = {
			'label': tf.FixedLenFeature([], tf.string),
			'img': tf.FixedLenFeature([], tf.string),
		})

	image = tf.decode_raw(features['img'],tf.uint8)

	image = tf.reshape(image,[96,96,1])

	label = tf.decode_raw(features['label'],tf.uint8)

	label = tf.reshape(label,[96*96*1])

	return image,label

def read_image_batch(file_queue, batch_size):

	img, label = readData_single(file_queue)

	min_after_dequeue = 2000

	capacity = 4000

	image_batch, label_batch = tf.train.shuffle_batch(
		tensors=[img, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)

	one_hot_labels = tf.reshape(label_batch, [1, 96, 96])

	return image_batch, one_hot_labels

	

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

			X = tf.nn.dropout(X,keep_prob =self.keep_prob)

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

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)


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


			X = tf.nn.dropout(X,keep_prob = self.keep_prob)

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

			X = tf.nn.dropout(X,keep_prob = self.keep_prob)

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

		with tf.name_scope('Gradient_Descent'):

			self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

	def train(self):

		train_file_path = "C:/Users/24400/Desktop/train_set_single_simple.tfrecords"

		train_image_filename_queue = tf.train.string_input_producer(string_tensor = tf.train.match_filenames_once(train_file_path),num_epochs = 1,shuffle = True)
		
		ckpt_path = "C:/Users/24400/Desktop/ckpt/model.ckpt"
		
		train_images,train_labels = read_image_batch(train_image_filename_queue,1)


		tf.summary.scalar("loss", self.loss_mean)
		
		tf.summary.scalar('accuracy', self.accurancy)
		
		merged_summary = tf.summary.merge_all()

		model_dir = "C:/Users/24400/Desktop/model"

		tb_dir = "C:/Users/24400/Desktop/logs"

		all_parameters_saver = tf.train.Saver()

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			
			sess.run(tf.local_variables_initializer())
			
			summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)
			
			tf.summary.FileWriter(model_dir, sess.graph)
			
			coord = tf.train.Coordinator()
			
			threads = tf.train.start_queue_runners(coord = coord)

			try:

				epoch = 1

				while not coord.should_stop():

					example,label = sess.run([train_images,train_labels])

					lo,acc,summary = sess.run([self.loss_mean,self.accurancy,merged_summary],feed_dict = {
							self.input_image:example,self.input_label:label,self.keep_prob:1.0,self.lamb:0.004
						})

					summary_writer.add_summary(summary, epoch)

					sess.run([self.train_step],feed_dict={
							self.input_image: example, self.input_label: label, self.keep_prob: 0.6,
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

def main():
	unet = Unet()
	unet.setup_network()
	unet.train()

main()
