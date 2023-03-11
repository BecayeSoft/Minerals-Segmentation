import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def show(img, title='', cmap=plt.cm.gray):
    plt.imshow(img, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()

def show_all(images, title=''):
    plt.figure(figsize=(12, 12))

    for i in range(0, len(images)):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.title(title)
    plt.show()

# def show_2_images(im1, im2, title1='', title2=''):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
#     ax1.imshow(im1, cmap=plt.cm.gray, interpolation='nearest')
#     ax1.axis('off')
#     ax1.set_title(title1)
#     ax2.imshow(im2, cmap=plt.cm.gray, interpolation='nearest')
#     ax2.set_title(title2)
#     ax2.axis('off')
#     plt.show()

def show_with_matplotlib(img, title=''):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR | GRAY image to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_RGB = img[:, :, ::-1]

    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_images(folder='image-samples'):
    """load the image-samples from folder """

    images = [cv2.imread(join(folder, file)) for file in listdir(folder) if isfile(join(folder, file))]

    return images

