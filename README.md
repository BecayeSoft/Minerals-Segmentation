# Minerals-Segmentation
Minerals segmentation with `scikit-image`.

The photos used to perform segmentation are from an optical microscope with an integrated HD camera. More precisely, they represent subparts of a sample (a single image) of rocks crushed into grains. Each grain represents a mineral (a mineral species).
The sample below has a size of 2.5 cm in diameter (weight of image weight ~2Go):

![Mashed Rock](https://github.com/BecayeSoft/Minerals-Segmentation/blob/main/images/mashed_rock.png)


Mission: in this practical exercise, the goal is to isolate (as well as possible) each grain. 
To do this, we will use sub-images of the photo in the previous figure. Each sub-image has a size of 600x600 pixels. 
Here is an example of a sub-image:

![Sample of mashed rock](https://github.com/BecayeSoft/Minerals-Segmentation/blob/main/images/mashed_rock_Sample.png)
