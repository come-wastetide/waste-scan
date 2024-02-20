from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from glob2 import glob
from sklearn.metrics import confusion_matrix
import torch

import pandas as pd
import numpy as np
import os
import zipfile as zf
import shutil
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import boto3
import io
from tqdm import tqdm

def list_keys(bucket_name,image_folder_key):

    '''

    input : bucket name & folder in which we want to list keys

    output : a list of keys without the name of the folder

    '''

    keys=[]

    s3=boto3.resource('s3')
    bucket=s3.Bucket(bucket_name)
    for i in bucket.objects.filter(Prefix=image_folder_key):
        keys.append(i.key)
    keys.pop(0) # remove the first element which is the folder name

    return keys


def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_value = arr[mid]

        if mid_value == target:
            return mid  # File found at index mid
        elif mid_value < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1  # File not found


def sort_images(bucket_name,image_folder_key,target_folder_name,excel_file_key):

    '''
    input : bucket name, folder in which the images are, folder in which we want to sort the images, excel file key

    output : images sorted in the target folder according to their category and subcategory
    
    '''

    keys = list_keys(bucket_name,image_folder_key)

    s3 = boto3.client('s3')


    excel_obj = s3.get_object(Bucket=bucket_name, Key=excel_file_key)
    excel_data = pd.read_excel(io.BytesIO(excel_obj['Body'].read()))

    # Get the total number of images to process
    total_images = len(keys)

    print(f"Sorting {total_images} images...")

    filenames = excel_data['PIC_NAME']
    filenames.dropna(inplace=True)
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_images, desc='Progress', unit='images')

    # Iterate through each row in the Excel file
    for key in keys:

        filename = key.split('/')[-1]

        # we are going to do a binary search to find the filename in the excel file

        index = binary_search(filenames, filename)
        
        i = index

        row = excel_data[index:index+1]
        category = row['WASTE_TYPE'][i]
        
        sub_category = row['WASTE_SUB_TYPE'][i]
        image_filename = row['PIC_NAME'][i] 
        image_key = image_folder_key + image_filename
        
        # Copy image from S3 to a new location with appropriate folder structure
        new_folder_key = f'{target_folder_name}/{category}/{sub_category}/'
        new_image_key = f'{new_folder_key}{image_filename}'

        #we check if the key exists in the bucket

        '''try:
            s3.head_object(Bucket=bucket_name, Key=image_key)
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"The image {image_key} does not exist in the bucket. Skipping...")
                progress_bar.update(1)
                continue'''

        '''# Create "category" folder if it does not exist
        try:
            s3.head_object(Bucket=bucket_name, Key=f'sorted_images/{category}/')
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3.put_object(Bucket=bucket_name, Key=f'sorted_images/{category}/', Body='')

        # Create "sub-category" folder if it does not exist

        try:
            s3.head_object(Bucket=bucket_name, Key=new_folder_key)
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3.put_object(Bucket=bucket_name, Key=new_folder_key, Body='')'''    

        # Copy image to new location
        s3.copy_object(
            Bucket=bucket_name,
            Key=new_image_key,
            CopySource={'Bucket': bucket_name, 'Key': image_key}
        )
        # Delete original image
        s3.delete_object(Bucket=bucket_name, Key=image_key)
            
        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print("Images sorted successfully.")

def is_sorted(arr):
    arr.dropna(inplace=True)
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            return False
    return True

def create_folder_names_from_excel(excel_data):

    '''

    input : excel_data with 'WASTE_TYPE' and 'WASTE_SUB_TYPE' in the keys

    output : folder_names 

    '''

    folder_names = []
    excel_data.dropna()

    for i in range(45080):
        category = excel_data['WASTE_TYPE'][i]
        sub_cat = excel_data['WASTE_SUB_TYPE'][i]

        folder_name = str(category) + '/' + str(sub_cat) + '/'
        if folder_name not in folder_names:
            folder_names.append(folder_name)

    return folder_names

def create_folders_if_needed(folder_name,folder_names):
    '''

    input : a list of folder names, with category & subcategory : 'Plastique/Autre déchet plastique/

    output : nothing but created the folders if not in the s3 bucket (within sorted images)

    ''' 

    for string in folder_names:
        category,sub_category = string.split('/')[0],string.split('/')[1]

        try:
            s3.head_object(Bucket=bucket_name, Key=f'{folder_name}/{category}/')
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3.put_object(Bucket=bucket_name, Key=f'{folder_name}/{category}/', Body='')

        # Create "sub-category" folder if it does not exist

        try:
            s3.head_object(Bucket=bucket_name, Key=f'{folder_name}/{category}/{sub_category}')
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3.put_object(Bucket=bucket_name, Key=f'{folder_name}/{category}/{sub_category}', Body='')

    print("Folders created successfully.")
