"""
	Description: Takes a file with data from training a model (loss, accuracy,
				 val_loss & val_accuracy) and displays on graphs
	Author: Jonas Pfefferman '24
	Date: 11/17/2023
"""

import json
import matplotlib.pyplot as plt
import argparse


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", required=True, help="Path to Data")
	args = vars(ap.parse_args())
	path = args["path"]

	with open(path, 'r') as modelInfo:
		info = json.load(modelInfo)

	# set up figures for the two losses
	loss = plt.figure(1)
	plt.plot(range(1, len(info["loss"])+1), info["loss"], 'b', label="training loss")
	plt.plot(range(1, len(info["val_loss"])+1), info["val_loss"], 'r', label="val_loss")
	plt.legend(["Training Loss", "val_loss"])
	plt.suptitle("Losses")
	plt.xlabel("Epochs")

	# set up figures for the two accuracies
	acc = plt.figure(2)
	plt.plot(range(1, len(info["accuracy"])+1), info["accuracy"], 'b', label='training accuracy')
	plt.plot(range(1, len(info["val_accuracy"])+1), info["val_accuracy"], 'r', label="val_accuracy")
	plt.legend(["Training Accuracy", "val_accuracy"])
	plt.suptitle("Accuracy")
	plt.xlabel("Epochs")
	plt.show()


main()