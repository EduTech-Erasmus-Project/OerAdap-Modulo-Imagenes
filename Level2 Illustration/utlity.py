import glob
import os
import pandas

root_path = '/media/edutech-pc06/Elements1/DataSet/ClasificacionPorContenido/PhotoChartDigital'
image_path_directory = dict()


def read_charts():
    for root, dirs, files in os.walk(root_path+'/Chart'):
        for file in files:
            image_path_directory[os.path.join(root, file)] = 0


def read_digital():
    for root, dirs, files in os.walk(root_path+'/Digital'):
        for file in files:
            if '.json' not in file:
                if '.html' not in file:
                    image_path_directory[os.path.join(root, file)] = 1


def read_photo():
    for root, dirs, files in os.walk(root_path+'/Photo'):
        for file in files:
            image_path_directory[os.path.join(root, file)] = 2


read_charts()
read_digital()
read_photo()

df = pandas.DataFrame.from_dict(image_path_directory.items())
df.columns = ['path', 'class']
df.to_csv('dataset.csv', index=False)
