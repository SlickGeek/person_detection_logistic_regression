import cv2
from random import shuffle
import glob
import h5py
import numpy as np

########################## first part: prepare data ###########################
shuffle_data = True

hdf5_path = './datasets/train_personvnonperson.hdf5'  # path to hdf5 file

# path to the images to train/test
person_nonperson_path = './images/persons_nonpersons/*.jpg'

addrs = glob.glob(person_nonperson_path)  # get path to each image
labels = [1 if 'pic' in addr[27:] else 0 for addr in addrs]  # label each image

if shuffle_data:
    c = list(zip(addrs, labels))  # bind each image with his label
    shuffle(c)

    # *c is used to separate all the tuples in the list c
    # "addrs" contains all the shuffled paths and
    # "labels" contains all the shuffled labels.
    (addrs, labels) = zip(*c)

# Divide the data into 80% for train and 20% for test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

##################### second part: create the h5py object #####################
train_shape = (len(train_addrs), 64, 64, 3)
test_shape = (len(test_addrs), 64, 64, 3)

# open a hdf5 file
f = h5py.File(hdf5_path, mode='w')

# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f.create_dataset("train_set_x", train_shape, np.uint8)
f.create_dataset("test_set_x", test_shape, np.uint8)

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key.
f.create_dataset("train_set_y", (len(train_addrs),), np.uint8)
f["train_set_y"][...] = train_labels

f.create_dataset("test_set_y", (len(test_addrs),), np.uint8)
f["test_set_y"][...] = test_labels

f.create_dataset("list_classes", (2,), dtype='|S10')
f["list_classes"][...] = [b'non-person', b'person']

######################## third part: write the images #########################

# loop over train paths
for i in range(len(train_addrs)):

    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr)
    # resize to (128,128)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["train_set_x"][i, ...] = img[None]

# loop over test paths
for i in range(len(test_addrs)):

    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    f["test_set_x"][i, ...] = img[None]

f.close()
