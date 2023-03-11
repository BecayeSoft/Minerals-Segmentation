import cv2
from ImageHelper import show, show_all, load_images
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.filters import sobel


images = load_images()
show_all(images, 'Sample Images')

# img = images[3]
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

images_threshes = []
images_edges = []
images_markers = []
images_segmentations = []
images_segmentation_filled = []
images_labels = []
images_from_labels = []

for img in images:

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. Threshold
    _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY )
    images_threshes.append(thresh)

    # 2. Detecting Edges
    edges = sobel(thresh)
    images_edges.append(edges)

    # 3. Markers
    # Mask: Set background to 1 and foreground to 2
    markers = np.zeros_like(thresh)
    markers[thresh == 0] = 1
    markers[thresh == 255] = 2
    images_markers.append(markers)

    # 4. Watershed
    # separate each mineral
    # When 2 minerals of the same class are neighbours, the might be mistaken for one  single mineral.
    # Watershed will separate every single the minerals.
    segmentation = watershed(edges, markers)
    images_segmentations.append(segmentation)

    # 5. filling black holes in minerals
    segmentation_filled = ndi.binary_fill_holes(segmentation - 1)
    images_segmentation_filled.append(segmentation_filled)

    # 6. Labels
    labels, _ = ndi.label(segmentation_filled)
    images_labels.append(labels)

    # 7. Image from labels
    labels_2_images = label2rgb(labels, image=img)
    images_from_labels.append(labels_2_images)

    # Plot
    # plt.contour(segmentation, [0.5], linewidths=1.2, colors='y')
    # show(labels_2_images, 'image_label_overlay')


show_all(images_threshes, 'Thresholds (Binary)')
show_all(images_edges, 'Images Edges (Sobel)')
show_all(images_markers, 'images_markers')
show_all(images_segmentations, 'Segmentation')
show_all(images_segmentation_filled, 'Segmentation with Filled Holes')

# Show with contour
show_all(images_from_labels, 'Colored images from labels')

