import os
import cv2
import pandas as pd


def open_image(idx, image):
    try:
        image = cv2.imread(image)
        cv2.resize(image, (10, 10))
    except Exception as e:
        print(f'Error while reading image: {image}, image index: {idx}')


def read_data():
    return pd.read_csv('dataframe.csv')


data = read_data()

for index, x in enumerate(data['path']):
    open_image(index, x)
