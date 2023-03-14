# Minerals-Segmentation
Minerals segmentation with `scikit-image`. <br/>
This is a school assignment to introduce us to the world of image segmentation using basic techniques. 
We did not spend a lot of time on it, to optimize the algorithm and get amazing results. 
The goal was to experiment with diverse techniques.

```
$ git clone https://github.com/BecayeSoft/Minerals-Segmentation/edit/main/README.md
$ python KMeans.py
$ python Watershed.py
```

The photos used to perform segmentation are from an optical microscope with an integrated HD camera. More precisely, they represent subparts of a sample (a single image) of rocks crushed into grains. Each grain represents a mineral (a mineral species).
The sample below has a size of 2.5 cm in diameter (weight of image weight ~2Go):

![Mashed Rock](https://github.com/BecayeSoft/Minerals-Segmentation/blob/main/images/mashed_rock.png)


Mission: in this practical exercise, the goal is to isolate (as well as possible) each grain. 
To do this, we will use sub-images of the photo in the previous figure. Each sub-image has a size of 600x600 pixels. 
Here is an example of a sub-image:

![Sample of mashed rock](https://github.com/BecayeSoft/Minerals-Segmentation/blob/main/images/mashed_rock_Sample.png)


## Algorithms used
Two methods have been tested in this assignment: K-Means Clustering and Watershed Segmentation.

### 1. K-Means Clustering
K_Means is a simple yet effective way to segment objects in an image.
Although there is a lot of room for improvement, it has proven to be better than the Watershed algorithm for our case.

### 2. Watershed Segmentation
The watershed has yielded poor results for our case. Again, this fast supposed to be a quick introduction to segmentation. 
So we did not spend a lot of time on it.
1. Threshold: A threshold is applied to improve edge detection.
2. Edge Detection: Sobel algorithm is applied to detect edges.
3. Markers: Markers are created from the threshold so that we can apply the "Watershed" algorithm.
4. Watershed: the watershed algorithm is applied to separate each mineral in the image.
5. Filling the holes: the holes in the segmented image are then filled.
6. Labelling: Finally, we extract labels from the image. Each label corresponds to a class of mineral.
