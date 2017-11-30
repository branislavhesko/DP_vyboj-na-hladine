from matplotlib import pyplot, rc
import numpy as np
import cv2


class DischargeLevel():


	def __init__(self, filename, threshold):
		self.filename = filename
		self.levels = []
		self.threshold = threshold

	def add_measurement_image(self, image):
		if len(image.shape) != 2:
			raise TypeError("Wrong image!")



	def get_level(self, image):
		bwimage = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
		level = [image[:,j].tolist().index(255) for j in range(image.shape[1])]

		return level
