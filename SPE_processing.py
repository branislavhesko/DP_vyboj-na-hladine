import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot, pylab
from SpeFileReader import SpeFileReader
import glob
from ImageFunctions import *

class SPE_processing:
    def __init__(self, path, experiment_number):
        self.i = experiment_number
        self.images = []
        #self.load_single_file(file_to_lood)
        self.images_multiple_files = False
        #self.calculate_mean_image()
        #self.show_images()
        #self.save_image("vysledok.png", self.mean_image, color = "jet")

    def delete_images(self):
        del self.images

    def load_single_file(self, f_name):
        """
        This method reads all the images in specififed file. No check if file exists!

        :param f_name: File to read
        :return:
        """
        self.images_multiple_files = False
        fid = SpeFileReader(f_name)
        is_end_file = False
        i = 0
        while not is_end_file:
            read_image = fid.load_img(i)
           #read_image = np.divide(read_image, 4)
           #read_image = read_image.astype(np.uint8)
           #cv2.equalizeHist(read_image,read_image)
            if len(read_image.shape) == 2:
               # if self.is_image_with_discharge(read_image):
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
        img = image
        pyplot.figure(1 , figsize= (20,10), dpi = 80)
        pyplot.imshow(img, clim=(np.mean(img), np.amax(cv2.blur(img,(31,31)))), cmap = "jet")
        """if i == -1 and image.shape:
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

        cv2.imwrite(file_name, final_image[300:700,300:700]) """

        pyplot.xlabel("horizontal view")
        pyplot.ylabel("vertical view")
        pyplot.savefig("vysledok" + (str(self.i)) + ".png")
        return 0

    def show_images(self, i = -1, image = []):
        if (i != -1):
            img = image
        else:
            img = self.images
        print(len(img))
        for image in img:
          print(image.shape)
          pyplot.figure(1 , figsize= (20,10), dpi = 80)
          pyplot.imshow(image, clim=(np.mean(image), np.amax(cv2.blur(image,(31,31)))), cmap = "jet")
          pyplot.xlabel("horizontal view")
          pyplot.ylabel("vertical view")
          pyplot.show()

    def calculate_mean_image(self):
        """
        Method, that calculates mean image from the set of the images

        :return:
        """
        self.mean_image = []
        #for i in range(len(self.images)):
            #self.images[i] = cv2.equalizeHist(self.images[i], self.images[i])

        self.mean_image = np.mean(self.images,0)

    def is_image_with_discharge(self, image):
        """
        Method, that decides whether image contains discharge or not

        :param image: Image to be analysed
        :return: True if image contains discharge, False otherwise
        """
        binary_image = cv2.threshold(image, 240, 0)
        binary_image = cv2.erode(binary_image, np.ones((3,3), dtype = np.uint8))
        val = np.mean(binary_image)
        print(val)
        if val >= 0.01:
            return True
        else:
            return False

    def calculate_mean_image_in_time(self):
        #TODO - verifiy functionality with a simple test
        self._mean_images = np.zeros((int(self._images_orig_file[0]), *self.images[0].shape))
        print(self._mean_images.shape)
        for i in range(self._images_orig_file[1]):
            self._mean_images[i,:,:] = np.mean(self.images[i::self._images_orig_file[0]], axis = 0)

    def get_mean_images(self):
        return self._mean_images

    def load_entire_folder(self, path):
        self.images_multiple_files = True
        self._images_orig_file = []
        files = glob.glob(path + "*.spe")
        print(files)
        if not len(files):
            raise FileNotFoundError("No files found")
        print(path)
        for file in files:
            fid = SpeFileReader(file)
            is_end_file = False
            i = 0
            while not is_end_file:
                read_image = fid.load_img(i)
                # read_image = np.divide(read_image, 4)
                # read_image = read_image.astype(np.uint8)
                # cv2.equalizeHist(read_image,read_image)
                if len(read_image.shape) == 2:
                    # if self.is_image_with_discharge(read_image):
                    self.images.append(read_image)
                    # imshow(self.images[-1], "float")
                else:
                    is_end_file = True
                i += 1
            self._images_orig_file.append(i)

        print("I have loaded " + str(len(files)) + " files.")
        print("Total number of images is " + str(len(self.images)))
        print(self._images_orig_file)
        print(len(self._images_orig_file))

        if not self._images_orig_file.count(self._images_orig_file[0]) == len(self._images_orig_file):
            raise ValueError("Images should be same length")

        return 1

    def get_images(self):
        return self.images

    def image_from_which_file(self):
        return self._images_orig_file

    def calculate_vertical_scan(self, images):
        if len(images.shape) != 3:
            raise TypeError("Images need to have 3 dimensions!")

        result_scan = np.zeros((images.shape[1], images.shape[0]), dtype = np.float64)

        for i in range(images.shape[0]):
            result_scan[:,i] = np.mean(images[i], 1)

        return result_scan

    def calculate_horizontal_scan(self, images):
        if len(images.shape) != 3:
            raise TypeError("Images need to have 3 dimensions!")

        result_scan = np.zeros((images.shape[0], images.shape[2]), dtype=np.float64)

        for i in range(images.shape[0]):
            result_scan[i, :] = np.mean(images[i], 0)

        return result_scan


if __name__ == "__main__":
    print("SPE images processing program loaded")
    path = glob.glob("./data/*.spe")
    print(path)
    for i in range(len(path)):
     spe = SPE_processing(path[i], i)
