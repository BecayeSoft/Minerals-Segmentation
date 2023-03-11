import cv2
from ImageHelper import show, show_all, load_images
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage import measure, io


images = load_images()
show_all(images, 'Sample Images')

img = images[3]
show(img, 'Original Image')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# --------------------------------------------------------------
# Threhsold
# separating background and foreground for better edge detection
# --------------------------------------------------------------

# thresh_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
# show(thresh_adapt, 'adaptative threshold')


# Thresholding
_, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY )
show(thresh, 'Threshold (Binary)')

# CLeaning up the image
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

show(eroded, 'Eroded')
show(dilated, 'Dilated')
show(img_gray * dilated, 'Masked Image')

# Mask
mask = (dilated == 255)
show(mask, 'Mask')

# Labeling
s = np.ones((3,3), np.uint8)
labeled_mask, n_labels = ndi.label(mask, structure=s)
show(labeled_mask, 'labeled_mask')

# Coloring the labels
colored_labels = label2rgb(labeled_mask)
show(colored_labels, 'img_colored')

# Clusters: Regions
# Get the regions properties for each label (grain)
regions = measure.regionprops(labeled_mask, img)
areas = []
for props in regions:
    areas.append(props.area)
np.reshape(img, 600, 600)
