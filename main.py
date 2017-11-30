import glob
import os
import sys
import SpeFileReader
import SPE_processing
from ImageFunctions import *




folder = "./data/"
filename = ""

subfolders = next(os.walk(folder))[1]


print(subfolders)
i = 0
for subfolder in subfolders:
    path = folder + subfolder + "/"
    files = glob.glob(path + "*.spe")
    print(files)
    spe_reader = SPE_processing.SPE_processing(path, i)
    i += 1
    spe_reader.load_entire_folder(path)
    spe_reader.calculate_mean_image_in_time()


    video_writing_folder = path + "video/"
    image_writing_folder = path + "images/"

    
    imsave(image_writing_folder+ subfolder +"_vertical_scan.png", spe_reader.calculate_vertical_scan(spe_reader.get_mean_images()), **{"resize": (1024,1024), "labels": ["time", "vertical mean", "vertical scan"]})
    imsave(image_writing_folder+ subfolder +"_horizontal_scan.png", spe_reader.calculate_horizontal_scan(spe_reader.get_mean_images()), **{"resize": (1024,1024),"labels": ["horizontal mean", "time", "horizontal scan"]})
    try:
        os.stat(image_writing_folder)
    except:
        os.mkdir(image_writing_folder)

    try:
        os.stat(video_writing_folder)
    except:
        os.mkdir(video_writing_folder)
    write_to_video(video_writing_folder + "mean_" + subfolder + ".avi",             spe_reader.get_mean_images())
    save_all_images(image_writing_folder+ subfolder, spe_reader.get_mean_images())
    