import numpy as np
from matplotlib import pyplot, rc
import cv2
import scipy
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage.measurements import center_of_mass
import math


def normalize(image):
    maxi = np.amax(image)
    mini = np.amin(image)
    im =(image - mini) / np.amax(image - mini)
    return np.abs(im - np.mean(im)) * 255

def imsave(filename, image, lower_upper_limit = (0,0), add_labels = False, color_bar = False, **kwargs):
    if not len(np.array(image).shape) == 2:
        raise IOError("Image not found")

    equalise = kwargs["equalise"] if "equalise" in kwargs else False
    show = kwargs["show"] if "show" in kwargs else False
    resize = kwargs["resize"] if "resize" in kwargs else False
    labels = kwargs["labels"] if "labels" in kwargs else ["x", "y", " "]
    if lower_upper_limit == (0,0):
        maximum = np.amax(cv2.blur(image,(5,5)))
        minimum = np.amin(cv2.blur(image,(5,5)))
    else:
        maximum = lower_upper_limit[1]
        minimum = lower_upper_limit[0]

    mean = np.mean(image)

    if resize:
        image = cv2.resize(image, resize)
    if equalise:
        image = cv2.equalizeHist(image)

    #FIXME - colorbar position
    #pyplot.figure(1, figsize=(16,9), dpi = 120)
    fig, ax = pyplot.subplots(**{"figsize": (16,9), "dpi":120})

    cax = ax.imshow(image, clim = (minimum,maximum), cmap = "nipy_spectral")
    if add_labels:
        centroids = center_of_mass(image, labels=image, index=range(np.amax(image)))
        for i in range(np.amax(image)):
            if not math.isnan(centroids[i][0]):
                ax.plot(centroids[i][1], centroids[i][0], "xg", markersize = 15)
                ax.text(centroids[i][1] + 5, centroids[i][0], str(i), fontsize=20, color='white')
    if color_bar:
        cbar = fig.colorbar(cax, ticks = [minimum, mean, maximum])
        cbar.ax.set_yticklabels([('min:' + str(minimum)), ('mean: ' + str(mean)), ('max: ' + str(maximum))])  # vertically oriented colorbar

    pyplot.xlabel(labels[0])
    pyplot.ylabel(labels[1])
    pyplot.title(labels[2])

    pyplot.savefig(filename)
    if show:
        pyplot.show()
    print(filename + str(" has been saved."))
    pyplot.close()



def imshow(image, lower_upper_limit = (0,0), **kwargs):
    pyplot.close()
    print(np.array(image).shape)

    if not len(np.array(image).shape) == 2:
        raise IOError("Image not found")

    equalise = kwargs["equalise"] if "equalise" in kwargs else False
    show = kwargs["show"] if "show" in kwargs else True
    resize = kwargs["resize"] if "resize" in kwargs else False
    labels = kwargs["labels"] if "labels" in kwargs else ["x", "y", " "]

    if resize:
        image = cv2.resize(image, resize)
    if equalise:
        image = cv2.equalizeHist(image)

    if lower_upper_limit == (0,0):
        maximum = np.amax(cv2.blur(image,(1,1)))
        minimum = np.amin(cv2.blur(image,(1,1)))
    else:
        maximum = lower_upper_limit[1]
        minimum = lower_upper_limit[0]
    mean = np.mean(image)

    # pyplot.figure(1, figsize=(16,9), dpi = 120)
    fig, ax = pyplot.subplots(**{"figsize": (10, 10), "dpi": 60})
    canvas = FigureCanvas(fig)
    cax = ax.imshow(image, clim = (minimum, maximum), cmap = "jet")

    cbar = fig.colorbar(cax, ticks = [minimum, mean, maximum])
    cbar.ax.set_yticklabels([('min:' + str(minimum)), ('mean: ' + str(mean)), ('max: ' + str(maximum))])  # vertically oriented colorbar

    pyplot.xlabel(labels[0])
    pyplot.ylabel(labels[1])
    pyplot.title(labels[2])

    if show:
        pyplot.show()


    return canvas

def change_image_type(image, option = "uint8"):
    if len(image.shape) != 2:
        print(image.shape)
        raise ValueError("Image has not valid size")


    if option == "uint8":
        if image.dtype == np.float64:
            im = image/4

        if image.dtype == np.uint16 or image.dtype == np.uint32:
            im = image/4

        return im.astype(np.uint8)

    if option  == "np.float64":
        im = image.astype(np.float64)
        if image.dtype == np.uint8:
            im = np.divide(im, 255)

        return im


def print_image_prop(image):
    print("------------------")
    print("IMAGE PROPERTIES: ")
    print("------------------")
    print("Size  | " + str(image.shape))
    print("Type  | " + str(image.dtype))
    print("Max   | " + str(np.amax(image)))
    print("Min   | " + str(np.amin(image)))
    print("Med   | " + str(np.median(image)))
    print("Mean  | " + str(np.mean(image)))



def pseudocolor_images(image, equalize):
    #if equalize:
    #    image = cv2.equalizeHist(image)
    print_image_prop(image)
    image = change_image_type(image, option = "uint8")
    return cv2.applyColorMap((image).astype(np.uint8), cv2.COLORMAP_JET)

def write_to_video(filename, images):
    #TODO - finish
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')


    video = cv2.VideoWriter(filename, fourcc, frameSize = (1920,1080), fps = 10, isColor=True)
    if not video.isOpened():
        print("Video not opened", file=sys.stderr)
        return -1
    limits = get_limits(images)
    for i in range(len(images)):
        canvas = imshow(images[i,:,:], limits)
        canvas.draw()
        im = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        im = im.reshape(1080,1920,3)
        print(im.shape)
        video.write(im[:,:,-1::-1])


    video.release()
    cv2.destroyAllWindows()

def get_limits(images):
    if not type(images) == np.ndarray:
        raise TypeError("Images only ndarray type accepted, not lists!")

    minimum = 1e9
    maximum = -1

    for i in range(len(images)):
        minimum = min(minimum, np.amin(cv2.blur(images[i,:,:],(35,35))))
        maximum = max(maximum, np.amax(cv2.blur(images[i,:,:],(35,35))))
    print((minimum,maximum))
    return (minimum, maximum)


def save_all_images(path,images):
    if not type(path) == str:
        raise TypeError("first argument string with path")

    lower_upper = get_limits(images)

    for index in range(len(images)):
        imsave(path + "_mean_" + str(index) + ".png", images[index,:,:], lower_upper_limit = lower_upper, **{"labels": ["x", "y", "skuska"], "show": False})

def plot_multiple_images(list_of_images):
    if len(list_of_images) == 2:
        fig, ax = pyplot.subplots(2,1)
    elif len(list_of_images) == 4:
        fig, ax = pyplot.subplots(2,2)
    else:
        print("NOT")
    #FIXME: add support for four images.
    for i,image in enumerate(list_of_images):
        ax[i%2].imshow(image, clim = (-10,10), cmap = "jet")

    pyplot.show()
if __name__ == "__main__":
    plot_multiple_images([np.zeros((10,10)), np.ones((10,10))])
