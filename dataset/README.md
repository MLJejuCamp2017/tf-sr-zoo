## Super resolution dataset collection.

## Introduction

There are some codes for downloading and preprocess 
the dataset which is used for single image super resolution.
The downloading code is from `https://github.com/phoenix104104/LapSRN`.
For more faster speed of reading data, we use tfrecord to feed to our 
model. And this is also beneficial for recording large dataset.

For single image super resolution, we used 91 images and 200 images 
from the training set of BSD to generate our training image patches.
Follow the method in (LapSRN)[http://graduatestudents.ucmerced.edu/wlai24/],
in each training batch, we randomly sample patches with size of 128x128.

## Setup
Use `python gen_tfrecord.py` to generate tfrecord data. If you want to change
dataset generate configurations, you can modify it in `options.py`.

To see more options, run `python gen_tfrecord.py --help`

