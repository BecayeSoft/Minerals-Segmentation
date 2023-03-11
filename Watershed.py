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

img = images[0]
show(img, 'Original Image')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# --------------------------------------------------------------
# Threshold
# separating background and foreground for better edge detection
# --------------------------------------------------------------
# thresh_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
# show(thresh_adapt, 'adaptative threshold')


# Thresholding
_, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY )
show(thresh, 'Threshold (Binary)')

# Detecting Edges
edges = sobel(thresh)
show(edges, 'Edges (Sobel)')

# --------------------------------------------------------------
# Histogram
# finding extreme values visually
# --------------------------------------------------------------
# hist = np.histogram(img_gray, bins=np.arange(0, 256))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# ax1.imshow(img_gray, cmap=plt.cm.gray, interpolation='nearest')
# ax1.axis('off')
# ax2.plot(hist[1][:-1], hist[0], lw=2)
# ax2.set_title('Grayscale Histogram')
# plt.show()
# print()

# --------------------------------------------------------------
# Markers
# markers of the background and the minerals
# background is set to 1 and foreground to 0
# --------------------------------------------------------------
markers = np.zeros_like(thresh)
markers[thresh == 0] = 1
markers[thresh == 255] = 2
# show_2_images(markers, thresh, 'Markers (1 or 2)', 'Thresh (0 or 255)')

# --------------------------------------------------------------
# Watershed Algorithm
# filling the edges (contours) with the markers (inside)
# --------------------------------------------------------------
segmentation = watershed(edges, markers)
show(segmentation, title='Watershed Segmentation')

# --------------------------------------------------------------
# Segmentation
# filling segmented image with colors that represent labels
# --------------------------------------------------------------
# fill black holes in the image (to reconstruct the mineral)
segmentation = ndi.binary_fill_holes(segmentation - 1)
show(segmentation, 'Segmentation with Filled Holes')

# label minerals from the segmented image
labels, _ = ndi.label(segmentation)

# convert labels into an image (each color represents a label)
labels_2_images = label2rgb(labels, image=img)

# draw minerals contours
plt.contour(segmentation, [0.5], linewidths=1.2, colors='y')
show(labels_2_images, 'image_label_overlay')


# Load the image
# Find the contours of the objects in the binary mask
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

object_imgs = []
# Loop through each contour and extract the corresponding object from the original image
for i, contour in enumerate(contours):
    # Create a bounding box around the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the image to the bounding box
    object_img = img[y:y+h, x:x+w]

    object_imgs.append(object_img)

plt.figure(figsize=(6, 6))

for i in range(0, 25):
    plt.subplot(10, 9, i + 1)
    plt.imshow(object_imgs[i])
    plt.axis('off')

plt.show()

minerals = []
for label in np.unique(labels):
    mask = labels == label
    extracted_object = np.zeros(img.shape, dtype="uint8")
    extracted_object[mask] = img[mask]

    show(extracted_object)
    minerals.append(extracted_object)
