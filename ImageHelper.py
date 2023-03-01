import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def show(img, title='', cmap=plt.cm.gray):
    plt.imshow(img, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()

def show_with_matplotlib(img, title=''):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR | GRAY image to RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_RGB = img[:, :, ::-1]

    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_images(folder='images'):
    """load the images from folder """

    images = [cv2.imread(join(folder, file)) for file in listdir(folder) if isfile(join(folder, file))]

    return images

