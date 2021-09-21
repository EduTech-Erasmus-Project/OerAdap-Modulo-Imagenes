import glob
import os
import pandas

root_path = '/media/edutech-pc06/Elements1/DataSet/ClasificacionPorContenido/'
parent_classes = [path for path in os.listdir(root_path) if '.' not in path]
current_class = None
image_path_dict = dict()


def read_formulas():
    formulas = os.listdir(root_path+'formula_images')
    for formula in formulas:
        image_path_dict[root_path+f'formula_images/{formula}'] = 0


def read_illustrations():
    parent = root_path+'Ilustraciones'
    illustrations = os.listdir(parent)

    for illustration in illustrations:
        children = os.listdir(parent+f'/{illustration}')
        for child in children:
            read_inner_classes(parent+f'/{illustration}/{child}')


def read_inner_classes(path):
    leaf = os.listdir(path)
    number_of_images = 0
    for leaf in leaf:
        if number_of_images < 63:
            image_path_dict[path+f'/{leaf}'] = 1
        number_of_images += 1


def read_tables():
    tables = os.listdir(root_path + 'Tablas')
    for table in tables:
        image_path_dict[root_path + f'Tablas/{table}'] = 2


read_formulas()
read_illustrations()
read_tables()

df = pandas.DataFrame.from_dict(image_path_dict.items())
df.columns = ['path', 'class']
df.to_csv(root_path+'dataframe.csv', index=False)
