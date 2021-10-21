import glob
import imghdr
import os
import pandas

root_path = '/home/edutech-pc06/Documentos/DATASETS/LVL1/Illustration'
image_path_directory = dict()


def read_charts():
    for root, dirs, files in os.walk(root_path+'/Chart'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = "a"
            else:
                print(f'Error while reading file: {file}')


def read_digital():
    for root, dirs, files in os.walk(root_path+'/Digital'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = "b"
            else:
                print(f'Error while reading file: {file}')


def read_photo():
    for root, dirs, files in os.walk(root_path+'/Photo'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = "c"
            else:
                print(f'Error while reading file: {file}')


read_charts()
read_digital()
read_photo()

df = pandas.DataFrame.from_dict(image_path_directory.items())
df.columns = ['path', 'class']
df.to_csv('dataframe.csv', index=False)
