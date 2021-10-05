import os
import pandas

root_path = '/home/edutech-pc06/Documentos/DATASETS/LVL1/'
image_path_directory = dict()


def read_equations():
    for root, dirs, files in os.walk(root_path+'/Equation'):
        for file in files:
            if '.json' not in file and '.html' not in file and '.py' not in file and '.md' not in file and '.pdf' not in file and '.txt' not in file:
                image_path_directory[os.path.join(root, file)] = 0


def read_illustration():
    for root, dirs, files in os.walk(root_path+'/Illustration'):
        for file in files:
            if '.json' not in file and '.html' not in file and '.py' not in file and '.md' not in file and '.pdf' not in file and '.txt' not in file:
                    image_path_directory[os.path.join(root, file)] = 1


def read_table():
    for root, dirs, files in os.walk(root_path+'/Table'):
        for file in files:
            if '.json' not in file and '.html' not in file and '.py' not in file and '.md' not in file and '.pdf' not in file and '.txt' not in file:
                image_path_directory[os.path.join(root, file)] = 2


read_equations()
read_illustration()
read_table()

df = pandas.DataFrame.from_dict(image_path_directory.items())
df.columns = ['path', 'class']
df.to_csv('dataframe.csv', index=False)