import imghdr
import os
import pandas

root_path = '/home/edutech-pc06/Documentos/DATASETS/LVL1/Illustration/Digital'
image_path_directory = dict()


def read_charts():
    for root, dirs, files in os.walk(root_path+'/Animated Illustration'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = 0
            else:
                print(f'Error while reading file: {file}, ignoring file.')


def read_digital():
    for root, dirs, files in os.walk(root_path+'/Logo'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = 1
            else:
                print(f'Error while reading file: {file}, ignoring file')


def read_photo():
    for root, dirs, files in os.walk(root_path+'/Screenshot'):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = 2
            else:
                print(f'Error while reading file: {file}, ignoring file')


read_charts()
read_digital()
read_photo()

df = pandas.DataFrame.from_dict(image_path_directory.items())
df.columns = ['path', 'class']
df.to_csv('dataframe.csv', index=False)
