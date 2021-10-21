import imghdr
import os
import pandas

root_path = '/home/edutech-pc06/Documentos/DATASETS/LVL1/Illustration/Digital/Animated Illustration'
image_path_directory = dict()


def read_images(sub_path, _class):
    print(f'............. WORKING WITH PATH: {sub_path} AND CLASS {_class}')
    for root, dirs, files in os.walk(root_path+'/'+sub_path):
        for file in files:
            if imghdr.what(os.path.join(root, file)) is not None:
                image_path_directory[os.path.join(root, file)] = _class
            else:
                print(f'Error while reading file: {file}, ignoring file.')


classes = {'Human': 'a', 'Anatomical': 'b', 'Vegetable': 'c', 'Transport': 'd', 'Structure': 'e', 'Food': 'f', 'Technology': 'g',
           'Animal': 'h', 'Tools': 'i', 'Object': 'j'}

for key in classes.keys():
    read_images(key, classes.get(key))

df = pandas.DataFrame.from_dict(image_path_directory.items())
df.columns = ['path', 'class']
df.to_csv('dataframe.csv', index=False)
