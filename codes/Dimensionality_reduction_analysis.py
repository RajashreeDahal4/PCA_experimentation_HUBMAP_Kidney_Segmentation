#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
with open("ev_individual_256.pkl",'rb') as pickle_file:
    individual_ev=pickle.load(pickle_file)
with open("ev_individual_length_256.pkl",'rb') as pickle_file:
    individual_len_ev=pickle.load(pickle_file)
    
    
with open("ev_batch_50_image.pkl",'rb') as pickle_file:
    batch_50_ev=pickle.load(pickle_file)
with open("ev_batch_50_length.pkl",'rb') as pickle_file:
    batch_50_len_ev=pickle.load(pickle_file)
    
    
with open("ev_batch_500.pkl",'rb') as pickle_file:
    batch_500_ev=pickle.load(pickle_file)
with open("ev_batch_500_length.pkl",'rb') as pickle_file:
    batch_500_len_ev=pickle.load(pickle_file)
    
#Lets make visualization plot for explained variance ratio for different pcs:
print(len(individual_ev))
print(len(batch_50_ev))
print(len(batch_500_ev))



# Flatten the 2D array of explained variance ratios into a 1D array
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
evs_flat = np.concatenate(individual_ev)
ax1.hist(evs_flat, bins=20)
ax1.set_xlabel('Explained Variance Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title("Histogram of Explained Variance Ratios")
# plt.hist(evs_flat, bins=20)

# len_evs_flat = np.concatenate(individual_len_ev)
ax2.hist(individual_len_ev, bins=20)
ax2.set_xlabel('Number of channels for explained variance ratio in each image')
ax2.set_ylabel('Frequency')
ax2.set_title("Histogram of number of features of Explained Variance Ratios")
plt.show()


# Flatten the 2D array of explained variance ratios into a 1D array
import numpy as np
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
evs_50_flat = np.concatenate(batch_50_ev)
ax1.hist(evs_50_flat, bins=20)
ax1.set_xlabel('Explained Variance Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title("Histogram of Explained Variance Ratios for batch size 50")

ax2.hist(batch_50_len_ev, bins=20)
ax2.set_xlabel('Number of channels for explained variance ratio with batch size of 50')
ax2.set_ylabel('Frequency')
ax2.set_title("Histogram of number of features of Explained Variance Ratios with batch size of 50")
plt.show()


import numpy as np
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
evs_500_flat = np.concatenate(batch_500_ev)
ax1.hist(evs_500_flat, bins=20)
ax1.set_xlabel('Explained Variance Ratio')
ax1.set_ylabel('Frequency')
ax1.set_title("Histogram of Explained Variance Ratios for batch size 500")

ax2.hist(batch_500_len_ev, bins=20)
ax2.set_xlabel('Number of channels for explained variance ratio with batch size of 500')
ax2.set_ylabel('Frequency')
ax2.set_title("Histogram of number of features of Explained Variance Ratios with batch size of 500")
plt.show()

