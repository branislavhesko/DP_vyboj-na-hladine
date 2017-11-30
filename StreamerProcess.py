import os
import sys
import glob
from SPE_processing import SPE_processing
from DetectStreamers import *
#TODO: finish
from ImageFunctions import *
from DataLog import DataLog
from StreamerProperties import StreamerProperties

def get_conductivity(folder):
        print(type(folder))
        index = str.find(folder, "m")
        conductiv = folder[:index]
        conductiv.replace("_",".")
        return float(conductiv)


path = "./images_orig/"
filename = "database.csv"

#conductivities = get_conductivities()
#log = DataLog("table.csv",["confuctivity", "time", "label_number", "width", "height", "number of pixels", "mean_intensity", "w_h_ratio"])

#folders = next(os.walk(path))[1]
#print(folders)


# files = glob.glob(path+"*.png")
# #conductivities = get_conductivity(str(folder))
#
# print(files)
# #spe_reader = SPE_processing(folder + files[0], 1)
# exper = 0
# i = 0
# for file in files:
#    exper += 1
#    conductivities = get_conductivity(file[14:])
#    print("Processing file " + file)
#    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#    # If streamer is not on the image.
#    #image = change_image_values_uint8(image)
#    #imshow(image)
#    result,bounds = process_image(image)
#    if type(result) == np.ndarray:
#        print_image_prop(result)
#        imsave(file[:-4] + "_i.jpg", result, add_labels = True)
#        for i in range(1,np.amax(result)):
#         if np.sum(result == i) > 0:
#             print(bounds)
#             props = StreamerProperties(image[bounds[0]:bounds[1], :])
#             #processed = np.array(result[result == i])
#             log.write_line(props.get_all_stats(result, i, conductivities, exper))
#
#
#
#
#
#
#
#
#
#
# exit(-1)







path = "./data/"
filename = "database.csv"

#conductivities = get_conductivities()
log = DataLog(filename,["confuctivity", "time", "label_number", "width", "height", "number of pixels", "mean_intensity", "w_h_ratio"])

folders = next(os.walk(path))[1]
print(folders)

for folder in folders:
    files = glob.glob(path+folder+"/*.spe")
    conductivities = get_conductivity(str(folder))
    print(files)
    spe_reader = SPE_processing(folder + files[0], 1)
    exper = 0
    for file in files:
       exper += 1
       print("Processing file " + file)
       spe_reader.load_single_file(file)
       i = 0
       for image in spe_reader.get_images():
    	# If streamer is not on the image.
            image = normalize(image)
            result,bounds = process_image(image)
            if type(result) == np.ndarray:
                props = StreamerProperties(image[bounds[0]:bounds[1], :])
                imsave("./results/" + file[:-4] + "_" + str(i) +".png", result, add_labels = True)
                log.write_line(props.get_all_stats(result, i, conductivities, exper))
            i+=1

       spe_reader.delete_images()








#
# path = "./data/"
# filename = "database.csv"

#conductivities = get_conductivities()




#
# folders = next(os.walk(path))[1]
# print(folders)
#
# minmax = []
# for folder in folders:
#     files = glob.glob(path+folder+"/*.spe")
#     conductivities = get_conductivity(str(folder))
#     print(path+folder)
#     spe_reader = SPE_processing(folder + files[0], 1)
#     spe_reader.load_entire_folder(path+ folder + "/")
#     limits = get_limits(np.array(spe_reader.get_images()[5:]))
#     minmax.append(limits)
#
#
#     spe_reader.delete_images()
# print(minmax)
#
#
