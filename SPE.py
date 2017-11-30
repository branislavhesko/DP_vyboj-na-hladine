import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot, pylab
from SpeFileReader import SpeFileReader
from ImageFunctions import *


class SPE:
    def __init__(self, file_to_lood):
        self.images = []
        self.load(file_to_lood)
        self.calculate_mean_image()

        self.save_image("vysledok.png", self.mean_image, color = "jet")

    def load(self, f_name):
        """
        This method reads all the images in specififed file. No check if file exists!

        :param f_name: File to read
        :return:
        """
        fid = SpeFileReader(f_name)
        is_end_file = False
        i = 0
        while not is_end_file:
            read_image = fid.load_img(i)
            read_image = np.divide(read_image, 4)
            read_image = read_image.astype(np.uint8)
            cv2.equalizeHist(read_image,read_image)
            if len(read_image.shape) == 2:
                if self.is_image_with_discharge(read_image):
                    self.images.append(read_image)
                    #imshow(self.images[-1], "float")
            else:
                is_end_file = True
            i += 1

    def save_image(self, file_name, image = [], i = -1, color = "gray"):
        """
        Method saving image into file

        :param file_name: Name of the file, also with suffix
        :param i: Index of the image to be save
        :param color: Apply pseudo-coloring? Possibility "hot" ,"jet"
        :param image: Image to be saved, if i remains default.
        :return: 0 if nothing happenned
        """

        if i == -1 and image.shape:
            img = image
        else:
            img = self.images[i]

        if np.dtype != np.uint8:
            img = img.astype(np.uint8)

        if color == "hot":
            final_image = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        elif color == "jet":
            final_image = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        else:
            final_image = img

        cv2.imwrite(file_name, final_image[300:700,300:700])
        return 0

    def show_images(self, i = -1, image = []):
        if (i == -1) and image.shape:
            img = image
        else:
            img = self.images[i]

        imshow(img, "float", option="equalise", name="example", color="jet", size=(640, 480))

    def calculate_mean_image(self):
        """
        Method, that calculates mean image from the set of the images

        :return:
        """
        self.mean_image = []
        for i in range(len(self.images)):
            self.images[i] = cv2.equalizeHist(self.images[i], self.images[i])

        self.mean_image = np.mean(self.images,0)

    def is_image_with_discharge(self, image):
        """
        Method, that decides whether image contains discharge or not

        :param image: Image to be analysed
        :return: True if image contains discharge, False otherwise
        """
        binary_image = threshold(image, 240, 0)
        binary_image = cv2.erode(binary_image, np.ones((3,3), dtype = np.uint8))
        val = np.mean(binary_image)
        print(val)
        if val >= 0.01:
            return True
        else:
            return False


if __name__ == "__main__":
    print("SPE images processing program loaded")
    spe = SPE("19_6kV_50_5us 2016 November 28 11_56_55.spe")