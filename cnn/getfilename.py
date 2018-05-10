import os
import numpy as np

files = os.listdir('test')

temp = []

for file in files:
    temp.append(file.replace('.jpg', ''))

np.save('filenames.npy', temp)
