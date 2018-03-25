import numpy as np  
import pandas as pd  
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import os
import datetime
import glob
import random
import sys


import matplotlib.pyplot as plt
import skimage.io 
import skimage.transform                               
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label                   

from keras.layers.merge import concatenate

from tensorflow.python.keras.models import Sequential,load_model, Model
from tensorflow.python.keras.layers import InputLayer, Input, Activation, BatchNormalization, Flatten
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.python.keras.layers import ELU,  ZeroPadding2D, UpSampling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,Dropout
from tensorflow.python.keras.layers import Concatenate, Cropping2D, Conv2DTranspose
from tensorflow.python.keras.layers import Concatenate, Add
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator











#import images and constants
IMG_HEIGHT, IMG_WIDTH = (256,256)  

train_directory = ('/.../input/stage1_train')
test_directory = ('/.../input/stage1_test')


train_ids = next(os.walk(train_directory))[1]
test_ids = next(os.walk(test_directory))[1]










# Get training data
def get_X_data(path, output_shape=(None, None)):
    '''
    Loads images from path/{id}/images/{id}.png into a numpy array
    '''
    img_paths = ['{0}/{1}/images/{1}.png'.format(path, id) for id in os.listdir(path)]
    X_data = np.array([skimage.transform.resize(skimage.io.imread(path)[:,:,:3], output_shape=output_shape, mode='constant', preserve_range=True) for path in img_paths], dtype=np.uint8)  #take only 3 channels/bands

    return X_data

X_train = get_X_data(train_directory, output_shape=(IMG_HEIGHT, IMG_WIDTH))
X_test = get_X_data(test_directory, output_shape=(IMG_HEIGHT, IMG_WIDTH))



def get_Y_data(path, output_shape=(None, None)):
    '''
    Loads and concatenates images from path/{id}/masks/{id}.png into a numpy array
    '''
    mask_paths = [glob.glob('{0}/{1}/masks/*.png'.format(path, mask_id)) for mask_id in os.listdir(path)]
    #2D array.  every mask that belongs to one image is in 1 list.  then we have a list of lists for each mask
    
    Y_data = []
    for i, img_masks in enumerate(mask_paths):  #loop through each individual nuclei for an image and combine them together
        #imread_collection reads in the list of masks from mask_paths that are enumerated. 
        #concatenate converts from <class 'skimage.io.collection.ImageCollection'> form to a <class 'numpy.ndarray'>
        masks = skimage.io.imread_collection(img_masks).concatenate()  #masks.shape = (num_masks, img_height, img_width)
        #np.max on axis=0 returns the maximum of each depth region in 3D?  wherever there is a nuclei detected in one of the masks, there is a higher value that's not 0, and it stores that?
        mask = np.max(masks, axis=0)                                   #mask.shape = (img_height, img_width)
        mask = skimage.transform.resize(mask, output_shape=output_shape+(1,), mode='constant', preserve_range=True)  #need to add an extra dimension so mask.shape = (img_height, img_width, 1)
        #append this truth mask to the array that stores everything
        Y_data.append(mask)
    Y_data = np.array(Y_data, dtype=np.bool)
    
    return Y_data
Y_train = get_Y_data(train_directory, output_shape=(IMG_HEIGHT, IMG_WIDTH))
#there are 670 masks and each mask is 256,256,1 so Y_train shape is 670,256,256,1.  1 is the channel
#Y_train is all booleans.  does this mean True for nuclei and false if it's background?




id = 64
skimage.io.imshow(X_train[id])
plt.show()
skimage.io.imshow(Y_train[id][:,:,0])
plt.show()







def get_unet(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
 #   u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
#    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
#    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
  #  u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[my_iou_metric])
    return model














#function is called by the fit function when model is running
def iou_metric(y_true_in, y_pred_in, print_table=False):
    #ground truth masks.  everything that has a value greater than 0.5 is in the array?  is this to exclude the background?
    #these go to high numbers beyond 255 as well
    labels = label(y_true_in > 0.5)
    #predicted masks
    y_pred = label(y_pred_in > 0.5)

    #label function labels all connected regions of an array.  we just did this in the lines above
    #above and stored it all in the labels array.  we now take all the unique labels in the labels
    #array, which gives us the actual number of nuclei
    true_objects = len(np.unique(labels))
    #this same process gives us the number of nuclei that our U-net architecture predicted there was.
    #later we can use watershed or something else to improve this.
    pred_objects = len(np.unique(y_pred))

    #intersection numbers are very large.  probably 0 for the background and larger numbers elsewhere depending on pixel strength?
    #check length of intersection array
    #shape of intersection is (true_objects,pred_objects)
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

 # Compute areas (needed for finding the union between all objects)
    #shape of area_true is (true_objects,)
    area_true = np.histogram(labels, bins = true_objects)[0]
    #shape of area_pred is (pred_objects,)
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    #shape of area_true becomes (true_objects,1) and area_pred becomes (1,pred_objects)
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)


    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

#we are feeding in the true mask and the predicted mask in batch sizes we have already selected before.  e.g. 32,256,256,1
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    print(y_true_in.shape,len(y_true_in))
    #metric is an array of 32 values.  one value for each image in the batch.  this network
    #is getting all zeros in the array which means predictions are not working or
    #network might not be learning
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value











#dice score (TP/(P+FP)) loss function for segmentation
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    print(type(y_true_f))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)









#optimizer = Adam(lr=1e-3)

optimizer = 'adam'
loss      = bce_dice_loss
metrics   = [my_iou_metric]


print('Loading model...')
model = get_unet()

datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

print('Fitting Augmenter')
datagen.fit(X_train)
print('Fitting model')
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=1)






print("Predicting")
# Predict on test data
test_mask = model.predict(X_test,verbose=1)




 
















