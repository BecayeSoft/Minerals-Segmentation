"""
 K-Means Clustering
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
from skimage.segmentation import slic

from ImageHelper import load_images, show


# Load the image
images = load_images()
img = images[1]
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
show(img, 'Original Image')

# Darken the image to remove noisy shades
M = np.ones(img.shape, dtype="uint8") * 30
darkened_img = cv.subtract(img, M)
show(darkened_img, 'Darkened Image')

# Apply the SLIC algorithm to segment the image
segments = slic(darkened_img, n_segments=36, compactness=70, max_size_factor=3)

minerals = []
channels_means = []
# Loop through each labeled region and extract the corresponding object from the original image
for label in np.unique(segments):
    mask = segments == label
    extracted_object = np.zeros(img.shape, dtype="uint8")
    extracted_object[mask] = img[mask]

    # show(extracted_object)
    minerals.append(extracted_object)

    b, g, r = cv.split(extracted_object)
    blue_mean = np.mean(b)
    green_mean = np.mean(g)
    red_mean = np.mean(r)
    channels_means.append([blue_mean, green_mean, red_mean])

# Save mean of each channel into a Dataframe
df_channels = pd.DataFrame(channels_means,
                   columns=['Blue Mean', 'Green Mean', 'Red Mean'])

# Plot the results
for i in range(0, 36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(minerals[i])
    plt.axis('off')

plt.show()
