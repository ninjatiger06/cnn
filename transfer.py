"""
	Description: Takes a pre-existing model and transfers it to train on a dataset
				 (in this case a deck of cards)
	Author: Jonas Pfefferman '24
	Date: 11/17/2023
"""

from __future__ import annotations
import tensorflow as tf
import tensorflow.keras.utils as utils
# import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.train import Checkpoint
import argparse
import json
import os

# def validateInput(prompt):
# 	while True:
# 		userInput = input(prompt)
# 		try:
# 			int(userInput)
# 			if userInput in [1, 2, 3]:
# 				return userInput
# 			else:
# 				print("Input must be an int between 1 and 3")
# 		except ValueError:
# 			print("Input must be an int between 1 and 3")

def createModel() -> keras.engine.functional.Functional:
	"""
	Purpose: Establishes a new model, including its inputs, outputs, losses, and optimizer
	Parameters: None
	Returns: The model as a model object
	"""
	netModel = vgg16.VGG16(
		include_top = True,
		weights = 'imagenet',
		classifier_activation = 'softmax',
	)

	print(f"netModel = {netModel}")

	netModel.trainable = False

	inputs = keras.Input(shape = (224, 224, 3))
	
	outputs = netModel(inputs)
	outputs = layers.Dense(53, activation = 'softmax')(outputs)

	optimizer = optimizers.legacy.Adam(learning_rate = 0.0001)
	loss = losses.CategoricalCrossentropy()

	model = keras.Model(inputs, outputs)

	model.compile(
		optimizer = optimizer,
		loss = loss,
		metrics = ['accuracy'],
	)

	return model

def evaluate(model: keras.engine.functional.Functional,
			train: tensorflow.python.data.ops.dataset_ops.MapDataset,
			validation: tensorflow.python.data.ops.dataset_ops.MapDataset,
			epochs: int, checkpointPath: str, infoPath: str):
	"""
	Purpose: Fits a given model to a given data set
	Parameters: The model (as a model object), training data (image dataset),
				validation (image dataset), number of epochs to run (int),
				where to save checkpoints (str of directory path), and where
				to save loss/accuracy data (str of path to json)
	Returns: The fitted model
	"""

	# would try keras.engine.functional.Functional as return hint but didn't get chance to check

	cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointPath,
													save_weights_only = True,
													verbose = 1)

	history = model.fit(
		train,
		batch_size = 32,
		epochs = epochs,
		verbose = 1,
		validation_data = validation,
		validation_batch_size = 32,
	)

	model.save(checkpointPath)
	
	pastHist = {
		"accuracy": [],
		"loss": [],
		"val_accuracy": [],
		"val_loss": []
	}

	try:
		with open(infoPath, 'r') as modelInfo:
			pastHist = json.load(modelInfo)
	except (FileNotFoundError, json.decoder.JSONDecodeError):
		pass
	
	if pastHist is not None:
		pastHist["accuracy"] += history.history["accuracy"]
		pastHist["loss"] += history.history["loss"]
		pastHist["val_accuracy"] += history.history["val_accuracy"]
		pastHist["val_loss"] += history.history["val_loss"]
	else:
		pastHist["accuracy"] = history.history["accuracy"]
		pastHist["loss"] = history.history["loss"]
		pastHist["val_accuracy"] = history.history["val_accuracy"]
		pastHist["val_loss"] = history.history["val_loss"]

	with open(infoPath, 'w') as modelInfo:
		json.dump(pastHist, modelInfo, indent=4)

	return history

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-l", "--load", required=False, help="Path to load save")
	ap.add_argument("-n", "--save", required=False, help="Path to where to save model")
	ap.add_argument("-p", "--data", required=True, help="Path to where to save model info")
	ap.add_argument("-e", "--epochs", required=True, help="Number of epochs for model to run")
	ap.add_argument("-m", "--modelPath", required=False, help="Model save path")
	args = vars(ap.parse_args())

	infoPath = args["data"]
	epochs = int(args["epochs"])

	train = utils.image_dataset_from_directory(
		'train',
		label_mode = 'categorical',
		image_size = (224, 224)
	)

	validation = utils.image_dataset_from_directory(
		'valid',
		label_mode = 'categorical',
		image_size = (224, 224)
	)

	print(train)
	print(validation)

	train = train.map(lambda x, y: (vgg16.preprocess_input(x), y))
	validation = validation.map(lambda x, y: (vgg16.preprocess_input(x), y))

	model = createModel()

	checkpoint = Checkpoint(model)

	# load previous weights if they exist
	if args["load"] != None:
		checkpointPath = args["load"]
		checkpointDir = os.path.dirname(checkpointPath)
		os.listdir(checkpointDir)

		checkpoint.restore(checkpointPath)
		model.load_weights(checkpointPath)

	# if not creates a new save
	else:
		checkpointPath = args["save"]

	history = evaluate(model, train, validation, epochs, checkpointPath, infoPath)

	# saving model itself
	if args["modelPath"] is not None:
		print(f"Saving model to {args['modelPath']}")
		model.save(args["modelPath"])


main()