# this function is used to resize the images to the desired size

import cv2
import os

def resize_images(input_folder, output_folder, size, filenames_to_resize = []):

    # filenames_to_resize is by default set to resize all the folders

    # we make sure that the output folder exists

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if filenames_to_resize==[]:
        n=len(os.listdir(input_folder))

        print(f'resizing {n} images')
        for filename in os.listdir(input_folder):
            img = cv2.imread(os.path.join(input_folder, filename))
            img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_folder, filename), img)

        print(f'successfully resized {n} images in {output_folder}')

    else : 
        n = len(filenames_to_resize)
        i=0
        print(f'resizing {n} images')
        available_files = os.listdir(input_folder)
        for filename in filenames_to_resize:
            if filename in available_files:
                img = cv2.imread(os.path.join(input_folder, filename))
                img = cv2.resize(img, size)
                cv2.imwrite(os.path.join(output_folder, filename), img)
                i+=1
        print(f'successfully resized {i} images in {output_folder} out of {n} filenames')




    # we loop through all the images in the input folder and resize them to the desired size
    


