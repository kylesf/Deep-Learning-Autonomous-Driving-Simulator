import numpy as np
import os
import sys
import pandas as pd
from skimage.io import imread
import cv2

def readDataIn():

	raw_data = pd.read_csv("./Data/driving_log.csv", header = 0)
	raw_data.columns = ('Center','Left','Right','Steering_Angle','Throttle','Brake','Speed')

	left = raw_data[['Left', 'Steering_Angle']].copy()
	left.loc[:, 'Steering_Angle'] += 0.16

	right = raw_data[['Right', 'Steering_Angle']].copy()
	right.loc[:, 'Steering_Angle'] -= 0.16

	img_paths = pd.concat([raw_data.Center, left.Left, right.Right]).str.strip()
	angles = pd.concat([raw_data.Steering_Angle, left.Steering_Angle, right.Steering_Angle])

	img_paths = img_paths.as_matrix()
	angles = angles.as_matrix()

	return img_paths,angles

def batch_generator(img_paths, angles, batch_size=128):
    len_data = img_paths.shape[0] * 2
    train_images = np.zeros((batch_size, 32, 64, 3))
    train_steering = np.zeros(batch_size)
    count = None
    while True:
        for j in range(64):
            if count is None or count >= len_data:
                count = 0
            idx = np.random.randint(img_paths.shape[0])
            train_images[j], train_steering[j], train_images[j+64], train_steering[j+64] = preprocessSinglePath(img_paths[idx],angles[idx])
            count += 2
        yield (train_images, train_steering)

def preprocessSinglePath(image_path,angle,flip = False):
	image = imread("./Data/"+image_path)
	image = image[32:135,0:320]
	resized = cv2.resize(image,(64,32))
	fresized = cv2.flip(resized, 1)
	fangle = angle * -1

	return resized, angle, fresized, fangle

def preprocessSingle(image):
	image = image[32:135,0:320]
	resized = cv2.resize(image,(64,32))
	return resized


