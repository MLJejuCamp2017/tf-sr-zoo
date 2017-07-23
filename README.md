# Single Image Super Resolution Model Zoo

## Project Introduction
This is a project for image super resolution. In this project, we 
implemented some state-of-art methods for image super resolution.
It contains the dataset preprocessing, model training, model evaluating.
Every subdir contains the codes for training and testing of one specific
method.

## Setup

### Prerequisites

* Tensorflow 1.1.1
* Linux with Nvidia GPU + cuDNN

### Geting Started
    git clone git@github.com:BingzheWu/tf-sr-zoo.git
    cd tf-sr-zoo
Then, chose the model you want to use, and get in the subdir to see more details.

### Some Results

### News
| Date | Update|
| -----| ------|
|20170720| Collect all trace of Evaluation
|20170715| Add attention Layer|
|20170710| Add Instance Normalization|
|20170708| Add evaluation module|
|20170705| Add gan based pix2pix model|
|20170703| Add LapSRN model |
|20170628| TFRecord Creator|
|20170624| Add Dataset Module|

## To-Do List

* Dataset Module (Done)
* Lap-SRN Method (Done)
* Pix2Pix Method (Done)
* Perceptual-GAN (Done)
* Instance Normalization(Done)
* Add unify testing module for every datasets (Doing)
* Add vis Tools (To Do)
* Combine L2 + L1 + adversarial loss with weighted term(To Do)
* Build a Docker Image (To Do)
