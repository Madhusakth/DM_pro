import numpy as np
import os
import csv

filenames = np.load('test_filename.npy')
print(filenames)

results = np.load('test_result.npy')

with open('submission.csv', 'w', newline='') as submissionfile:
    writer = csv.writer(submissionfile)
    writer.writerow(['id', 'landmarks'])
    print(len(filenames))
    print(len(results))
    for index, filename in enumerate(filenames):
        writer.writerow([filename, '%s %s' % (results[index], str(1))])
