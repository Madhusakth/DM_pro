import csv
import operator
import numpy as np
from PIL import Image
import os, os.path
from sklearn.model_selection import train_test_split

preprocessedPath = 'images/preprocessed'
arr = []
inputs = []
train_Y = []

print('Loading Images')
csvfile = open('train.csv', 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
data = sorted(data, key=operator.itemgetter(0))
i = 0
kk = 0

for filename in os.listdir(preprocessedPath):
    arr.extend(np.load(os.path.join(preprocessedPath, filename)))

path = 'images/training'
for entry in data:
    if int(entry[2]) > 450 or int(entry[2]) < 0:
        i = i + 1
	kk = kk + 1
        continue
    try:
        filename = entry[0] + '.jpg'
        im = Image.open(os.path.join(path, filename))
        inputs.append(arr[kk])
        train_Y.append(int(entry[2]))
        i = i + 1
        kk = kk + 1

    except Exception as e:
        print('Can not identify file %s' %(filename))
        i = i + 1

train_X = np.array(inputs)
train_X = train_X.reshape(-1, 2048, 1, 1)
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
