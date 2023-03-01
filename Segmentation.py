import cv2
from ImageHelper import show, load_images
import matplotlib.pyplot as plt
import numpy as np


from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.filters import sobel

images = load_images()
img = images[1]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show(img, 'Orginal Image')

# --------------------------
# Region-based Segmentation
# --------------------------

thresh_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
# show(thresh_adapt, 'adaptative threshold')

_, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY )
show(thresh, 'Binary threshold')

elevation_map = sobel(thresh_adapt)
show(elevation_map, 'elevation_map')

markers = np.zeros_like(thresh)
markers[thresh < 30] = 1
markers[thresh > 150] = 2
show(markers, 'markers')

# ------------------------------
segmentation = watershed(elevation_map, markers)
show(segmentation, title='segmentation')


# ------------------------------

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_coins, image=img)         # color-coded labels image

plt.contour(segmentation, [0.5], linewidths=1.2, colors='y')
show(img)
show(image_label_overlay, 'image_label_overlay')