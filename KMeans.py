import cv2 as cv

# ---------------------
#  K-Means Clustering

import numpy as np
from ImageHelper import load_images, show
from skimage.segmentation import slic

# Load the image
images = load_images()
img = images[1]
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
show(img, 'Original')

# Darken the image to remove noisy shades
M = np.ones(img.shape, dtype="uint8") * 30
darker_img = cv.subtract(img, M)
show(darker_img, 'Darker Image')

# CLeaning up the image
# kernel = np.ones((7,7), np.uint8)
# eroded = cv.erode(img, kernel, iterations=1)
# dilated = cv.dilate(eroded, kernel, iterations=1)
#
# show(img, 'Original')
# show(eroded, 'Eroded')
# show(dilated, 'Dilated')
# show(img * dilated, 'Masked Image')




# Apply the SLIC algorithm to segment the image
# segments = slic(img, n_segments=100, compactness=10)
segments = slic(light_image, n_segments=36, compactness=70)
# segments = slic(img)

minerals = []
# Loop through each labeled region and extract the corresponding object from the original image
for label in np.unique(segments):
    mask = segments == label
    extracted_object = np.zeros(img.shape, dtype="uint8")
    extracted_object[mask] = img[mask]

    show(extracted_object)
    minerals.append(extracted_object)

# show(img)
# show(minerals[1])

grain = minerals[20]
b, g, r = cv.split(grain)
