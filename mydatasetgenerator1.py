# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 10:43:29 2018

@author: qwert
"""
import cv2
import os
import numpy as np
import tqdm
from random import shuffle
def process_test_data(TEST_DIR, IMG_SIZE):
    testing_data = []
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data