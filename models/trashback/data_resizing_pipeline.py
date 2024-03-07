# this function is used to resize the images to the desired size

import cv2
import os

def resize_images(input_folder, output_folder, size):

    # we make sure that the output folder exists

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # we loop through all the images in the input folder and resize them to the desired size
    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        img = cv2.resize(img, size)
        cv2.imwrite(os.path.join(output_folder, filename), img)
    return

