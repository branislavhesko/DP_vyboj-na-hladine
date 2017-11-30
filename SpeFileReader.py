import cv2
import numpy as np
import sys
import time
from matplotlib import pylab
import os
from ImageFunctions import imshow, write_to_video


class SpeFileReader:

    def __init__(self, filename):
        self._file = open(filename, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.uint16)[0])
        self._ydim = np.int64(self.read_at(656,1,np.uint16)[0])
        print("Image Xdimension: " + str(self._xdim) + " Ydimension " + str(self._ydim))
        print(np.int64(self.read_at(108, 1, np.uint16)[0]))

        self._image_dimension = self._xdim * self._ydim * 2
        value = self._file.seek(0,os.SEEK_END)
        self._file_size = self._file.tell()

    def read_at(self, pos, size, ntype):
        self._file.seek(pos)
        return np.fromfile(self._file, ntype, size)

    def load_img(self, image_number):
        stride = self._image_dimension * (image_number) + 8*(image_number)
        if (self._file_size - stride - 4096) < self._image_dimension:
            print(self._file_size - stride - 256)
            return np.empty(1)
        else:
            img = self.read_at((4100 + stride), self._xdim*self._ydim, np.uint16)
            return img.reshape((self._ydim,self._xdim))/16

    def _load_date_time(self):
        rawdate = self.read_at(20, 9,np.int8)
        rawtime = self.read_at(172, 6,np.int8)
        strdate = ''

        for ch in rawdate:
            strdate += chr(ch)

        for ch in rawtime:
            strdate += chr(ch)

        self._date_time = time.strptime(strdate, "%d%b%Y%H%M%S")
        print(self._date_time)

    def close(self):
        self._file.close()



if __name__ == "__main__":
    print("Testing...")
    fid = SpeFileReader("./data/00.spe")
    images = []
    is_end_file = False
    i = 0
    while not is_end_file:
        read_image = fid.load_img(i)
        # read_image = np.divide(read_image, 4)
        # read_image = read_image.astype(np.uint8)
        # cv2.equalizeHist(read_image,read_image)
        if len(read_image.shape) == 2:
            # if self.is_image_with_discharge(read_image):
            images.append(read_image)
            #imshow(images[-1], **{"show": True})
        else:
            is_end_file = True
        i += 1

    write_to_video("skuska.avi", np.array(images[:]))