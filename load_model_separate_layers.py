from pprint import pprint

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
import tensorflow.keras.preprocessing as preprocessing
import tensorflow.keras.callbacks as callbacks
from tensorflow.data import Dataset
from tensorflow.train import Checkpoint
import tensorflow as tf
import numpy as np
import cv2

SZ = 239

train, validation = utils.image_dataset_from_directory(
	'keptPokemon',
	label_mode = 'categorical',
	batch_size = 256,
	image_size = (239, 239),
	seed = 69,
	validation_split = 0.15,
	subset = 'both'
)


class_names = train.class_names

print("Class Names:")
pprint(class_names)

# If you are loading from a checkpoint, you need to define the model
# before loading.  If you are loading a saved model, you can comment out
# the entire class definition because the model architecture is built 
# into the saved model.

class Net():
	def __init__(self, image_size):
		self.model = tf.keras.Sequential()
		# depth, frame size are first 2 args
		# First layer of a Sequential Model should get input_shape as arg
		# Input: 239 x 239 x 3
		self.model.add(layers.Conv2D(
			16,
			11, 
			strides=2, 
			activation=activations.relu,
			input_shape=input_size,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 115 x 115 x 3
		self.model.add(layers.Conv2D(
			16, 
			3, 
			strides=1, 
			activation=activations.relu,
			input_shape=input_size,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 113 x 113 x 3
		self.model.add(layers.Conv2D(
			18, 
			3, 
			strides=1, 
			activation=activations.relu,
			input_shape=input_size,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 111 x 111 x 3
		self.model.add(layers.Conv2D(
			18, 
			3, 
			strides=1, 
			activation=activations.relu,
			input_shape=input_size,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 109 x 109 x 3
		self.model.add(layers.MaxPool2D(
			pool_size=3,
			strides=2,
		))
		tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None,)
		# Size: 54 x 54 x 16
		self.model.add(layers.Conv2D(
			32,
			3,
			strides=1,
			activation=activations.relu,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 52 x 52 x 32
		self.model.add(layers.Conv2D(
			32,
			2,
			strides=1,
			activation=activations.relu,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# Size: 60 x 60 x 64
		# if run into memory issues cut to 32
		self.model.add(layers.Conv2D(
			64,
			3,
			strides=1,
			activation=activations.relu,
			kernel_regularizer=tf.keras.regularizers.L2(),
		))
		tf.keras.layers.BatchNormalization()
		# # Size: 58 x 58
		self.model.add(layers.MaxPool2D(
			pool_size=2,
			strides=2,
		))
		# Size: 29 x 29 x 64
		self.model.add(layers.Flatten())
		# Size: 53824
		# if NAN is loss value, model is too big
		self.model.add(layers.Dense(4096, activation=activations.relu))
		tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None,)
		self.model.add(layers.Dense(1024, activation=activations.relu))
		tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None,)
		self.model.add(layers.Dense(512, activation=activations.relu))
		# Size of last Dense layer MUST match # of classes
		self.model.add(layers.Dense(21, activation=activations.softmax))
		self.optimizer = optimizers.Adam(learning_rate=0.00001)
		self.loss = losses.CategoricalCrossentropy()
		self.model.compile(
			loss = self.loss,
			optimizer = self.optimizer,
			metrics = ['accuracy'],
		)
	def __str__(self):
		self.model.summary()
		return ""

tf.get_logger().setLevel('FATAL')
for pokemon in class_names:
	# Get the first image of that pokemon and set it up
	img = cv2.imread(f'keptPokemon/{pokemon}/{pokemon}1.jpg')
	img = cv2.resize(img, (SZ, SZ))
	img = utils.img_to_array(img)
	img = img[tf.newaxis, ...]

	# Did checkpoints every 2 epochs up to 40.
	#   or every 4 epochs up to 80 or every 8 up to 200 or every 10 to 380.
	for k in range(40, 321, 40):
		# Set up the architecture and load in the checkpoint weights
		net = Net((SZ, SZ, 3))
		# print(net)
		checkpoint = Checkpoint(net.model)
		checkpoint.restore(f'checkpoints/checkpoint_5_{k:02d}').expect_partial()
		# Get the first conv layer, feed the image and set it up for viewing
		filters = net.model.layers[0](img)[0]
		shape = filters.shape
		filters = filters.numpy()
		# Put all filters in one big mosaic image with 2 rows, padded by 
		#   20px gray strips.  
		# Scaling up the filters by 3x to make them easier to see
		cols = shape[2] // 2
		mosaic = np.zeros(
			(6*shape[0] + 20, 3*cols*shape[1] + (cols - 1)*20)
		)  
		# Print the filter max and average to screen so we can see how much
		#   the classification uses this filter.
		print(f'{pokemon:>12} Chkpt {k:03d} Maxes:', end = ' ')
		second_str = '                       Avgs: '
		# Shape[2] = number of filters
		for i in range(shape[2]):
			# Get just one filter
			filter = filters[0:shape[0],0:shape[1],i]
			# Calculate and print max and avg
			maxes = []
			avgs = []
			for j in range(shape[0]):
				maxes.append(max(filter[j]))
				avgs.append(sum(filter[j])/len(filter[j]))
			print(f'{max(maxes):8.4f}', end = ' ')
			second_str += f'{sum(avgs)/len(avgs):9.4f}'
			# Triple the filter size to make it easier to see
			filter = cv2.resize(filter, (3*shape[0], 3*shape[1]))
			# Rescale so the grayscale is more useful
			if max(maxes) > 0:
				filter = filter / max(maxes) * 2
			else:
				filter = 0
			# Locate the filter in the mosaic and copy the values in
			offset = ((i % 2)*(3*shape[0] + 20), (i // 2)*(3*shape[1] + 20))
			mosaic[
				offset[0]:offset[0] + 3*shape[0], 
				offset[1]:offset[1] + 3*shape[1]] = filter  
		print()
		print(f'{second_str}')
		# Make the gray stripes that separate the filters
		# Vertical Stripes
		for i in range(1, cols):
			start_vert_stripe = 3*i*shape[1] + (i - 1)*20
			mosaic[
				0:mosaic.shape[0], 
				start_vert_stripe:start_vert_stripe + 20] = np.ones(
					(mosaic.shape[0], 20)) * 0.5  
		# Horizontal Stripe
		mosaic[3*shape[0]:3*shape[0] + 20, 0:mosaic.shape[1]] = np.ones(
				(20, mosaic.shape[1])) * 0.5
		# Display the image
		cv2.imshow(f'{pokemon} Checkpoint {k}', mosaic)
		if chr(cv2.waitKey(0)) == 'q':
			quit()
		cv2.destroyAllWindows()
