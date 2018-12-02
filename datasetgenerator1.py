import cv2
import numpy as np
from os import listdir
from os.path import join
from random import shuffle

class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    def label_generator(self):
        labels = []
        for filename in listdir(self.data_dir):
            token = filename.split(".")
            if token[-1] == "jpg":
                labels.append(token[0])
                #print(token[0]
        return labels
    def training_data_generator(self):
        for filename in listdir(self.data_dir):
            path =

