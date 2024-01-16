
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

# Use the first GPU available
count=0
with tf.device('/GPU:0'):
    images = []
    evs=[]
    ev_len=[]
    s=0
    train_x=sorted(glob("train_256/*"))
    train_y=sorted(glob("masks_256/*"))
        # Convert the images 
    for i,img_path in enumerate(train_x):
        images_flat=[]
        img = Image.open(img_path)
        img=np.array(img)
        img = Image.fromarray(img)
        img=np.array(img)
        img_flat=img.flatten()


        # Perform PCA
        pca = PCA(n_components=0.95,svd_solver='full')
        X_pca = pca.fit_transform(img_flat.reshape(-1,1))
        ev=pca.explained_variance_ratio_
        evs.append(ev)
        ev_len.append(len(ev))
        
            # Reshape the reduced images back into their original shape
        X_reduced = pca.inverse_transform(X_pca).reshape(img.shape)
        images_reduced = tf.zeros([256, 256, 3])
        images_reduced = tf.reshape(X_reduced, [256, 256,3])

            # Convert the numpy array to a PIL image
        imgs = Image.fromarray(images_reduced.numpy().astype(np.uint8))
        # Load the images as NumPy arrays
        image1=imgs
        image2 = plt.imread(img_path)
        image3 = plt.imread(train_y[i])

#         Save the image
        img_path = os.path.join('individual_256/', f"{i+1}.jpg")
        imgs.save(img_path)
        mask_path = os.path.join('individual_masks_256/',f"{i+1}.jpg")
        plt.imsave(mask_path,image3)
        print(f"Saved reduced image {i+1}/{len(images)}")

        
        #save the pickle files for PCs visualization
with open('ev_individual_256.pkl', 'wb') as f:
    pickle.dump(evs, f)

with open('ev_individual_length_256.pkl', 'wb') as f:
    pickle.dump(ev_len, f)

