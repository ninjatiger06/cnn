import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.train import Checkpoint
import tensorflow.data as data
from os import listdir
import os.path

# Skipping getting data for now

# VERSION 1: Class-based model
class Model:
	def __init__(self, input_size):
		self.model = tf.keras.Sequential()
		# depth, frame size are first 2 args
		# First layer of a Sequential Model should get input_shape as arg
		# Input: 239 x 239 x 3
		self.model.add(layers.Conv2D(
			12, 
			11, 
			strides=2, 
			activation=activations.relu,
			input_shape=input_size,
		))
		# Size: 115 x 115 x 3
		self.model.add(layers.Conv2D(
			12, 
			11, 
			strides=2, 
			activation=activations.relu,
			input_shape=input_size,
		))
		tf.keras.layers.BatchNormalization()
		# Size: 53 x 53 x 3
		self.model.add(layers.MaxPool2D(
			pool_size=3,
			strides=2,
		))
		tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None,)
		# Size: 26 x 26 x 16
		self.model.add(layers.Conv2D(
			16,
			4,
			strides=2,
			activation=activations.relu,
		))
		# Size: 12 x 12 x 32
		self.model.add(layers.Conv2D(
			32,
			2,
			strides=1,
			activation=activations.relu,
		))
		# Size: 11 x 11 x 64
		self.model.add(layers.Conv2D(
			64,
			1,
			strides=1,
			activation=activations.relu,
		))
		# # Size: 11 x 11
		# self.model.add(layers.MaxPool2D(
		# 	pool_size=2,
		# 	strides=1,
		# ))
		# Size: 11 x 11 x 64
		self.model.add(layers.Flatten())
		# Size: 7744
		self.model.add(layers.Dense(4096, activation=activations.relu))
		self.model.add(layers.Dense(1024, activation=activations.relu))
		self.model.add(layers.Dense(512, activation=activations.relu))
		# Size of last Dense layer MUST match # of classes
		self.model.add(layers.Dense(151, activation=activations.softmax))
		self.optimizer = optimizers.Adam(learning_rate=0.0001)
		self.loss = losses.CategoricalCrossentropy()
		self.model.compile(
			loss = self.loss,
			optimizer = self.optimizer,
			metrics = ['accuracy'],
		)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = Model((239, 239, 3))
model.model.summary()

save_path = "model/"
plotHistoryPath = "modelHistory.json"

train, validation = utils.image_dataset_from_directory(
	'PokemonData',
	label_mode = 'categorical',
	batch_size = 256,
	image_size = (239, 239),
	seed = 69,
	validation_split = 0.15,
	subset = 'both'
)

# train = utils.image_dataset_from_directory(
# 	'train',
# 	label_mode = 'categorical',
# 	image_size = (239, 239)
# )

# validation = utils.image_dataset_from_directory(
# 	'valid',
# 	label_mode = 'categorical',
# 	image_size = (239, 239)
# )

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
validation = validation.cache().prefetch(buffer_size = data.AUTOTUNE)

# load previous weights if they exist
model.model.load_weights(save_path)

cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath = save_path, save_weights_only = True, verbose = 1)

history = model.model.fit(
	train,
	batch_size = 256,
	epochs = 670,
	verbose = 1,
	validation_data = validation,
	validation_batch_size = 32
)

print(f"Saving model to {save_path}")
model.model.save(save_path)

import json
print(f"Saving training history to {plotHistoryPath}")

old_history = {
	"accuracy": [],
	"loss": [],
	"val_accuracy": [],
	"val_loss": [],
}
try:
	with open(plotHistoryPath, "r") as f:
		old_history = json.load(f)
except (FileNotFoundError, json.decoder.JSONDecodeError):
	pass

if old_history is not None:
	old_history["accuracy"] += history.history["accuracy"]
	old_history["loss"] += history.history["loss"]
	old_history["val_accuracy"] += history.history["val_accuracy"]
	old_history["val_loss"] += history.history["val_loss"]

with open(plotHistoryPath, "w") as f:
	json.dump(old_history, f, indent=4)