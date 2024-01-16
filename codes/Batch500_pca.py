
import os
from glob import glob
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import pickle
import math


# Use the first GPU available
count=0
with tf.device('/GPU:0'):
    batch_size=500
    train_x=sorted(glob("train_256/*"))
    train_y=sorted(glob("masks_256/*"))
    num_batches=math.ceil(len(train_x)/batch_size)    
    count=0
    evs=[]
    ev_length=[]
    for i in range(num_batches):
        images = []
        masks= []
        masks_flat = []
        images_flat = []
        s=0
        for img_path in train_x[count:count+batch_size]:
            count=count+1
            img = Image.open(img_path)
            images.append(np.array(img))
    
            # Convert the images 
        for img in images:
            img = Image.fromarray(img)
            img=np.array(img)
            img_flat=img.flatten()
            images_flat.append(img_flat)

        # Perform PCA
        pca = PCA(n_components=0.95,svd_solver='full')
        X_pca = pca.fit_transform(images_flat)
         # Do something with X_pca
        ev=pca.explained_variance_ratio_
        evs.append(ev)
        ev_length.append(len(ev))
            # Reshape the reduced images back into their original shape
        X_reduced = pca.inverse_transform(X_pca)
        images_reduced = tf.zeros([len(images_flat), 256, 256, 3])
        images_reduced = tf.reshape(X_reduced, [len(images_flat), 256, 256,3])
        # Assuming images_reduced is a list of reduced images
        for j, imgs in enumerate(images_reduced):
            # Convert the numpy array to a PIL image
            imgs = Image.fromarray(imgs.numpy().astype(np.uint8))
            # Load the images as NumPy arrays
            image1=imgs
            image2 = plt.imread(train_x[i*batch_size+j])
            image3 = plt.imread(train_y[i*batch_size+j])
     #Save the image
            img_path = os.path.join('batch_500_256/', f"{i*500+j+1}.jpg")
            imgs.save(img_path)

            

with open('ev_batch_500.pkl', 'wb') as f:
    pickle.dump(evs, f)

with open('ev_batch_500_length.pkl', 'wb') as f:
    pickle.dump(ev_length, f)






