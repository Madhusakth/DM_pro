import csv
import numpy as np
from PIL import Image
import os, os.path
from sklearn.model_selection import train_test_split

numSamples = {}
topClasses = []
inputs = []
train_Y = []

print('Preprocessing Images')
csvfile = open('C:/Users/rimra/Documents/Spring 2018/Data Mining/Project/train.csv', 'r')
csvreader = csv.reader(csvfile)
data = [line[:3] for line in csvreader]
data = data[1:]
i = 0

for entry in data:
    classVal = int(entry[2])
    numSamples[classVal] = numSamples.get(classVal, 0) + 1

temp = sorted(numSamples, key=numSamples.get, reverse=True)
topClasses = temp[:50]
print(topClasses)

path = 'C:/Users/rimra/Documents/Spring 2018/Data Mining/Project/training-images'
#print(os.listdir(path))
for filename in os.listdir(path):
    entry = int(data[i][2])
    if entry not in topClasses:
        i = i + 1
        continue

    try:
        im = Image.open(os.path.join(path, filename)).convert('L')
        im_data = list(im.getdata())
        #arr = np.array(im)
        #arr = arr[:,:,0]
        #arr = arr.astype('float16')
        #arr = arr / 255.
        inputs.append(im_data)
        train_Y.append(entry)
        i = i + 1
    except Exception as e:
        print('Can not identify file %s' %(filename))
        i = i + 1

train_X = np.array(inputs)
#print(train_X.shape)
#train_X = train_X.reshape(-1, 256, 256, 1)

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
#print(train_X.shape)
#print(valid_X.shape)
print('Getting PCA')
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca = pca.fit(train_X)
X_pca_train = pca.transform(train_X)
X_pca_valid = pca.transform(valid_X)

print('Dumping')
from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(X_pca_train,train_label,'t150.input', multilabel=False)

print('Finished')
dump_svmlight_file(X_pca_valid,valid_label,'v150.input', multilabel=False)
