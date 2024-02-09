"""
	Description: Takes a number of images and uses the model to predict what it is
	Author: Jonas Pfefferman, based off of Milan Kumar's code
	Date: 12/7/23
"""
import cv2
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image prediction set")
ap.add_argument("-m", "--model", required=True, help="Path to model")
args = vars(ap.parse_args())

classes = os.listdir("./test/")

images = []
image = args["image"]
if os.path.isdir(image):
	contents = os.listdir(image)
	for img in contents:
		# image = prepImage(cv2.imread(img))
		img = args["image"] + "\\" + img
		image = cv2.imread(img)
		image = vgg16.preprocess_input(image)
		images.append(image)
else:
	# image = prepImage(cv2.imread(image))
	image = cv2.imread(image)
	image = vgg16.preprocess_input(image)
	images.append(image)

#------------------------------------------------------------------------------#
import tensorflow as tf
import tensorflow.keras.models as models

model = models.load_model(args["model"])

for image in images:
	image = tf.expand_dims(image, 0)
	prediction = list(model.predict(image)[0])
	predictions_with_labels = list(zip(prediction, classes))
	predictions_with_labels.sort(reverse=True, key=lambda x: x[0])

	print("\n\nResults:")
	for confidence, label in predictions_with_labels[:4]:
		print(f"{label : >8}: {confidence * 100:.2f}%")