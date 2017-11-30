"""Summary

Attributes:
    eroded (list): Description
"""
import numpy as np
import cv2
from matplotlib import pyplot
from ImageFunctions import *
import skimage.io
import skimage.morphology
from matplotlib import rc
from skimage.filters.rank import gradient
import os
import sys
import math
from scipy.signal import medfilt
from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass

# os.listdir("./")

from Exceptions import *

#TODO: add support for multilable comparison instead of 2 labels?
def border_length_between_labels(image, labels, connectivity = 4):
    """Short summary.

    Parameters
    ----------
    image : type
        labeled image, which is analysed
    labels : type
        two element tuple

    Returns
    -------
    array
        an array of points representing the border of two labels...

    Raises
    -------
    ExceptionName
        Why the exception is raised.

    """
    border = []
    x,y = (image == labels[0]).nonzero()
    for i in range(len(x)):
        if connectivity == 4:
            points = ((image[x-1, y]), (image[x, y-1]), (image[x+1, y]), (image[x, y+1]))
        elif connectivity == 8:
            points = ((image[x+1, y-1]), (image[x-1, y+1]), (image[x+1, y+1]), (image[x-1, y-1]), (image[x-1, y]), (image[x, y-1]), (image[x+1, y]), (image[x, y+1]))

        if labels[1] in points:
            border.append(image[x[i], y[i]])

    return border

#TODO: finish method
def merge_labels_touching_by_border(labeled_image, centroids):
    for center in centroids:
        for center2 in centroids:
            if abs(center[1][0] - center2[1][0]) < 100 and center != center2:
                border = border_length_between_labels(labeled_image, (center[0], center2[0]))
                max_x, max_y = get_border_direction(border=border)
                # Calculate angle of the border?
                if max_x < max_y:
                    #Objects are the same
                    labeled_image[labeled_image == center2[0]] = center[0]

    return labeled_image



def split_distinct_discharges(labeled_image, centroids):
    for centroid in centroids:
        temp_image = labeled_image[labeled_image == centroid[0]]
        temp_image = skimage.measure.label(temp_image, background=0)
        heights = []
        if np.amax(temp_image) > 1:
            for i in range(np.amax(tem_image)):
                heights.append(get_height(image, i))

            max_height = np.amax(np.array(heights))
            for i, height in enumerate(heights):
                if height != max_height and height > 0.3*max_height:
                    labeled_image[labeled_image = i] = centroids[-1][0] + 1


    return labeled_image


def get_height(image, label_number):
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
			break


	return bottom_row - top_row




def remove_nan_centroids(centroids):
    new_centroids = []
    for centroid in centroids:
        if not math.isnan(centroid[0]) and not math.isnan(centroid[1][0]):
            new_centroids.append(centroid)


    return tuple(new_centroids)

def get_border_direction(border):
    """Get maximal difference in x and y position in border pixels.

    Parameters
    ----------
    border : list/tuple
        List of border pixels.

    Returns
    -------
    tuple
        max x difference and max y difference
    """
    val_x = border[0][0]
    vyl_y = border[0][1]
    max_x = 0
    max_y = 0
    for x,y in border:
        max_x = max(x - val_x, max_x)
        max_y = max(y - val_y, max_y)

    return (max_x, max_y)

#TODO: add comments
def detect_streamers_watershed(image):
    """Summary

    Args:
        image (TYPE): Description

    Raises:
        IOError: Description
    """

    if type(image) != np.ndarray:
        raise IOError("One of the images is not ndarray type")

    image_blurred = blur_and_threshold_image(image)
    distance_image = ndi.distance_transform_edt(image_blurred)
    seeds = cv2.bitwise_and(cv2.threshold(image, 3*np.mean(image), 255, cv2.THRESH_BINARY)[1], image_blurred)
    seeds = cv2.erode(seeds, np.ones((3,3)))
    label = skimage.morphology.label(seeds, neighbors=4)
    #imshow(seeds)
    watershed_im = skimage.morphology.watershed(-distance_image, label, mask = image_blurred)
    #print(watershed_im)
    watershed_im =  remove_small_objects(watershed_im)

    #imshow(watershed_im)
    return watershed_im

def vertical_projection(image):
    """Summary

    Args:
        image (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        IOError: Description
    """
    if type(image) != np.ndarray:
        raise IOError("Image is not ndarray type")

    vertical_proj_array = np.zeros(image.shape[0])
    vertical_proj_array = np.mean(image, axis=1)
    return vertical_proj_array


#FIXME: mean value?
def detect_ROI(image):
    """
    docstring here
            :param image:

    Args:
        image (TYPE): Description

    Returns:
        TYPE: Description
    """
    vertical_proj_array = medfilt(vertical_projection(image), 11)
    mean_value = np.mean(vertical_proj_array)
    poz_max = np.argmax(vertical_proj_array)
    value = vertical_proj_array[poz_max]
    i = 0

    while value > mean_value:
        i += 1
        value = vertical_proj_array[poz_max - i]

    lower_limit = poz_max - i

    i = 0
    value = vertical_proj_array[poz_max]

    while value > mean_value:
        i += 1
        if poz_max+i < image.shape[0]:
            value = vertical_proj_array[poz_max + i]
        else:
            break
    upper_limit = poz_max + i

    # Extend interval by some values
    upper_limit = upper_limit * 1.1
    lower_limit = lower_limit * 0.9

    return (int(lower_limit), int(upper_limit))

eroded = []
def is_streamer_on_image(image):
    """Summary

    Args:
        images (TYPE): Description

    Returns:
        TYPE: Description
    """
    eroded_value = np.sum(cv2.erode(cv2.threshold(
        image, 2 * np.mean(image), 1, type=cv2.THRESH_BINARY)[1], np.ones((7, 7))))
    eroded.append(eroded_value)
    if eroded_value > 5:
        print("Streamer is on image")

        return True
    else:
        print("Streamer is NOT on image")

        return False


def horizontal_projection(image):
    """Summary

    Args:
        image (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        IOError: Description
    """
    if type(image) != np.ndarray:
        raise TypeError("Image is not ndarray type")

    horizontal_proj_array = np.zeros(image.shape[1])
    horizontal_proj_array = np.mean(image, axis=0)
    return horizontal_proj_array


#TODO: make it better, also height could play role, not only centers and implement somewhat how much are objects touching

def remove_small_objects(labeled_image):
    if type(labeled_image) != np.ndarray:
        raise TypeError("Fuck it")

    centers = center_of_mass(labeled_image, labels = labeled_image, index=range(np.amax(labeled_image)))
    for i in range(1,len(centers)):
        centers[i] = (i, centers[i])



    for i in range(1,len(centers)):
        for j in range(1,len(centers)):
            #print(centers[i][1][0])
            if abs(centers[i][1][0] - centers[j][1][0]) > 8 and abs(centers[i][1][1]-
                centers[j][1][1]) < 30:
                label_max = max(centers[i][0], centers[j][0])
                label_min = min(centers[i][0], centers[j][0])
                labeled_image[labeled_image == label_max] = label_min


    for i in range(np.amax(labeled_image)):
        if np.sum(labeled_image == i) < 200:
            labeled_image[labeled_image == i] = 0

    centers = center_of_mass(labeled_image, labels = labeled_image, index=range(np.amax(labeled_image)))
    for i in range(1,len(centers)):
        centers[i] = (i, centers[i])

    print(centers)


    return labeled_image

def experiments():
    """Summary
    """
    image = cv2.cvtColor(cv2.imread(
        "images_orig/0_1ms_9KV19.png"), cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float64)
    image = cv2.normalize(image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_64F)
    image_orig = np.copy(image)
    image[image < 0.0649] = 0
    image = cv2.blur(image, (15, 15))
    image[image > 1e-5] = 1
    fig, ax = pyplot.subplots(1, 2)
    bounds = detect_ROI(image_orig)
    #image = image[bounds[0]:bounds[1]]
    ax[0].imshow(image_orig, clim=(0, np.amax(cv2.blur(image, (5, 5)))))
    ax[1].plot(medfilt(vertical_projection(image_orig), 5)[-1::-1],np.arange(1024,0,-1), "-r", linewidth=3)
    ax[1].plot(np.ones(image_orig.shape[0]) *
               np.mean(vertical_projection(image_orig)), np.arange(1024,0,-1), "-b")
    pyplot.gca().invert_yaxis()

    ax[1].set_xlabel("mean intensity")
    ax[1].set_ylabel("row")
    pyplot.show()



def blur_and_threshold_image(image):
    """Summary

    Args:
        image (TYPE): Description

    Returns:
        TYPE: Description
    """
    image_blurred = cv2.GaussianBlur(image, (11,11),0)
    _,image_blurred = cv2.threshold(image_blurred, 17, 255, cv2.THRESH_BINARY)
    image_blurred = cv2.dilate(image_blurred, np.ones([1,1]))
    #imshow(image_blurred)
    return image_blurred


def process_image(image):
    if not is_streamer_on_image(image):
        return False,0
    image = image.astype(np.float64)
    image = cv2.normalize(image, dst = image, dtype = cv2.CV_64F)
    #FIXME: detect_roi returns positions, not image!
    bounds = detect_ROI(image)
    image = image[bounds[0]:bounds[1], :]
    image = detect_streamers_watershed(image)

    return image,bounds

#FIXME: structuring element triangle form!
if __name__ == "__main__":
    means = []
    ifa = []
    for i in range(0, 72):
        image = cv2.cvtColor(cv2.imread(
            str(i) + ".png"), cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float64)
        if is_streamer_on_image(image):
            bounds = detect_ROI(image)
            image = image[bounds[0]:bounds[1], :]
            res = detect_streamers_watershed(image)
            #print((res == 1).nonzero())
            imsave( "skuska_a_" + str(i) + ".png", res, add_labels =True)

            #raise StarWarsException("End")


#TODO: implement two thresholds, one for centers, OTSU?, another one weaker, for merging objects.ins
