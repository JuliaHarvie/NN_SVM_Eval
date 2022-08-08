import numpy as np
import cv2
import argparse
import glob
import random
import os
import sys
import pandas as pd
 
### Set up arguments, can accept path to the directory where the images and their masks are stores as 
#   as well as size of the patches the image will be subdivided into 

argparser = argparse.ArgumentParser() 
argparser.add_argument("filename",
		help="Path to folder containing the images.")
argparser.add_argument("patch_size",
		help="Number of pixels of one side of the square that will be used to subdivide the images", type = int)
argparser.add_argument("image_extension",
		help="Extension of images to be analysed. Ex png, jpg")
      
args = argparser.parse_args(sys.argv[1:])
path = args.filename
window_size = args.patch_size
extension = args.image_extension

### Reading in the images using imread from open CV and storing them image arrays in a list

image = []
mask = []
for file in sorted(glob.glob(path + "/*." + extension)):
    if "Mask" in file:
        mask.append(cv2.imread(file, 0))
    else:
        image.append(cv2.imread(file))


### Subdivide the image into patches based off of the size specified in the orginal arguments. 
#   Each patch will be condensed into a single row of an array. One array will be created per image 
#   and these arrays will be stored in a list to be used a single fold for the straification 10-fold 
#   cross validation.

folds = []
count = 0
for im in image:
    print(f"Subdividing image: {count}")
# Divide to normalize    
    im_mask = mask[count]/255 
# Error in mask data, not perfect binary, this code here
# corrects for this
    for x in range(0,im_mask.size):
        if np.ravel(im_mask)[x] <= 0.5:
            np.ravel(im_mask)[x] = 0
        elif np.ravel(im_mask)[x] > 0.5:
            np.ravel(im_mask)[x] = 1

# Convert to LUV colour space and isolate the lummonosity channel
    im = cv2.cvtColor(im, cv2.COLOR_BGR2Luv)
    im_l = cv2.split(im)[0]

#Index the central pixels for all the patches in this image
    col_len = im.shape[0]
    row_len = im.shape[1]
    start = row_len*int(window_size/2) + int(window_size/2)
    stop = (start + row_len) 
    patch_index = np.arange(start,stop,window_size)
    row_folds = len(patch_index)
    i = 1
    while i < col_len/window_size:
        new = patch_index[-(row_folds):]+(row_len*window_size)
        patch_index = np.append(patch_index,new)
        i+=1

### For each central pixel extract its label form the mask image and the luminosity values
#   for all of the pixels contained in the patch.
    fold = np.zeros((patch_index.size, (window_size ** 2)+2),dtype=int)   
    n = 0
    for PI in patch_index:      
        patch = np.zeros((1,(window_size ** 2)+2), dtype=int)
        pixels = []
        for r in range(PI - (row_len*int(window_size/2)), PI + (row_len*int(window_size/2)) + 1, row_len):
            for c in range(r-int(window_size/2),r+int(window_size/2)+1):
                pixels.append(c)
        fold[n, 0] = np.ravel(im_mask)[PI,]
        fold[n, 1] = PI
        fold[n, 2:(window_size ** 2)+2] = np.ravel(im_l)[pixels]
        n+=1
#Iterate through for all the images, storing the arrays in the list folds to access later         
    count+=1
    folds.append(fold)

### Create the directory the folds will be outputted to
if not os.path.exists("Datasets"):
	os.mkdir("Datasets")

### Now that the folds have been defined they need to be stored into testing, validation and 
#   training data.
for f in range(0,len(image)):
    print(f'Exporting crossvalidation: {f+1}')
    test = folds[f]
# Define remaining folds that training and validation can be selected from 
    seq = [x for x in range(0,len(folds)) if x != f]
# 3 folds assigned to validation, reaminign 6 training 
    val = np.random.choice(seq, 3, False)
    train = []
    [train.append(x) for x in seq if x not in val]

# Exporting to csv    
    testing_data = folds[f]
    pd.DataFrame(testing_data).to_csv(f'Datasets/Fold_{f+1}_testing.csv', index=False)
    
    training_data = np.empty((1,folds[f].shape[1]))
    for x in train:
        training_data = np.append(training_data,folds[x],axis=0)
    training_data = np.delete(training_data,0,0)
    pd.DataFrame(training_data).to_csv(f'Datasets/Fold_{f+1}_training.csv', index=False)

    validation_data = np.empty((1,folds[f].shape[1]))
    for x in val:
        validation_data = np.append(validation_data,folds[x],axis=0)
    validation_data = np.delete(validation_data,0,0)
    pd.DataFrame(validation_data).to_csv(f'Datasets/Fold_{f+1}_validation.csv', index=False)


