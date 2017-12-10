import numpy as np
import cv2
import matplotlib
import skimage
import sklearn
from scipy.ndimage.measurements import center_of_mass


#TODO: include more params
#TODO: include center of mass,
#FIXME: include time in period
#FIXME: check whether data_format does not cuts floats
class StreamerProperties:
	def __init__(self, orig_image):
		self._orig_image = orig_image


	def get_width(self, image, label_number):
		left = False
		right_col = image.shape[1]
		left_col = 0
		for i in range(image.shape[1]):
			if np.sum(image[:,i] == label_number) > 0 and not left:
				left_col = i
				left = True

			if left and np.sum(image[:,i] == label_number) == 0:
				right_col = i

		return right_col - left_col

	def get_height(self,image, label_number):
		top = False
		bottom = False
		bottom_row = image.shape[0]
		top_row = 0
		for i in range(image.shape[0]):
			if np.sum(image[i,:] == label_number) > 0 and not top:
				top_row = i
				top = True
			if top and np.sum(image[i,:] == label_number) == 0:
				bottom_row = i


		return bottom_row - top_row

	def get_number_of_pixels(self,image,label_number):
		return np.sum(image == label_number)


	def get_mean_intensity(self,image, label_number, orig_image):
		return np.sum(orig_image[image == label_number])

	def get_width_to_height_ratio(self, width, height):
		return height / width

	def get_centroid_position(self, orig_image, label_number):
		image_on_label = orig_image[orig_image == label_number]
		print(center_of_mass(image_on_label, labels=image_on_label, index = 1))


	def get_number_of_branches(self, orig_image, label_number):
		pass

	def get_x_position(self, orig_image, label_number, bottom_row):
		pass

	def get_all_stats(self, image, label_number, conductivity, time):
		width = self.get_width(image,label_number)
		height = self.get_height(image, label_number)
		number_pix = self.get_number_of_pixels(image, label_number)
		mean_intens = self.get_mean_intensity(image, label_number, self._orig_image)
		w_h_ratio = self.get_width_to_height_ratio(width,height)



		return (conductivity, time, label_number, width, height, number_pix, mean_intens, w_h_ratio)
