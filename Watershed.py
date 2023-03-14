import cv2
from ImageHelper import show, show_2_images, load_images
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.filters import sobel


images = load_images()
img = images[0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Darken the image
M = np.ones(img.shape, dtype="uint8") * 30
img = cv2.subtract(img, M)

show(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Original Image')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# --------------------------------------------------------------
# Threshold
# separating background and foreground for better edge detection
# --------------------------------------------------------------
_, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY )
show(thresh, 'Threshold (Binary)')

# Detecting Edges
edges = sobel(thresh)
show(edges, 'Edges (Sobel)')

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

minerals = []
for label in np.unique(labels):
    mask = labels == label
    extracted_object = np.zeros(img.shape, dtype="uint8")
    extracted_object[mask] = img[mask]

    minerals.append(extracted_object)


# Plot the results
for i in range(0, 36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(minerals[i])
    plt.axis('off')

plt.show()


show_2_images(img, labels_2_images, 'Original Image', 'Labeled Image')