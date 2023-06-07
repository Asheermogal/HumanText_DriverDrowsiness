import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from collections import Counter
from PIL import Image
from itertools import groupby


''''
here in this example we need to read the text document , and use the text document to generate the path 
for the model . for all the ok startus we are going to generate the text. 
'''
file_path = r"E:\Data Science Project\Humantext Recognition\Datasets\words_new.txt"
with open(file_path) as f:
    lines = f.readlines()

label_raw=lines

image_texts  = []
image_paths  = []
default_path = "E:/Data Science Project/Humantext Recognition/Datasets/iam_words/words"
for label in label_raw:
  if label.split()[1]=="ok":
    image_texts.append(label.split()[-1])
    image_paths.append(default_path+"/"+label.split()[0].split("-")[0]+"/"+label.split()[0].split("-")[0]+"-"+label.split()[0].split("-")[1]+"/"+label.split()[0]+".png")


image_texts=image_texts
image_paths=image_paths

corrupt_images = []

for path in image_paths:
    try:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    except:
        corrupt_images.append(path)

for path in corrupt_images:
    
    corrupt_index = image_paths.index(path)
    del image_paths[corrupt_index]
    del image_texts[corrupt_index]

### get vocabulary for the current dataset
vocab = set("".join(map(str, image_texts)))
print(sorted(vocab))

max_label_len = max([len(str(text)) for text in image_texts])
max_label_len


char_list = sorted(vocab)

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
    
    return pad_sequences([dig_lst], maxlen=max_label_len, padding='post', value=len(char_list))[0]


padded_image_texts = list(map(encode_to_labels, image_texts))

padded_image_texts[0]