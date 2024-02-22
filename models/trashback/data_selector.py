import os
import openpyxl
import boto3
import pandas as pd
import io
from data_processing import list_keys

keys = list_keys(bucket_name, train_folder_key)
keys_test = list_keys(bucket_name, test_folder_key)

def initialize_dict_of_categories(categories):

    dict_of_categories = {}

    for category in categories:
        dict_of_categories[category] = []
    
    return dict_of_categories

def process_data(keys, dict_of_categories, categories):

    # gather the data that exists in the s3 bucket

    for key in keys:
        filename = key.split('/')[-1]
        category, sub_category = key.split('/')[-3], key.split('/')[-2]
        '''dict_of_images[key] = category + '/' + sub_category
        dict_filenames[filename] = category'''
        if category in categories:
            dict_of_categories[category].append(filename)

bucket_name = 'trashback-data'
test_folder_key = 'sorted_images_test/'
train_folder_key = 'sorted_images/'
excel_file_key = 'waste_pics.xlsx'


def get_excel_data(excel_file_key, bucket_name='trashback-data'):
    s3 = boto3.client('s3')

    excel_obj = s3.get_object(Bucket=bucket_name, Key=excel_file_key)
    excel_data = pd.read_excel(io.BytesIO(excel_obj['Body'].read()))
    excel_data.dropna(inplace=True)
    return excel_data

excel_data = get_excel_data(excel_file_key)

categories = excel_data['WASTE_TYPE'].unique()


dict_of_categories = initialize_dict_of_categories(categories)

process_data(keys, dict_of_categories, categories)
process_data(keys_test, dict_of_categories, categories)
