
# Objectif de ce code
(Ce code est indépendant de tout le reste du dépôt)

À partir d'une vidéo d'une personne source et d'une autre d'une personne, notre objectif est de générer une nouvelle vidéo de la cible effectuant les mêmes mouvements que la source. 

[Allez voir le sujet du TP ici](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

# Posture-Guided Image Synthesis Project

## Overview
This project implements posture-guided image synthesis, transferring motion from a source video to a target person. Inspired by ["Everybody Dance Now"](https://arxiv.org/abs/1808.07371) by Chan et al. (ICCV 2019), it includes three approaches for generating images from skeleton input: (1) nearest-neighbor matching, (2) a direct neural network, and (3) a GAN-based approach, all built using PyTorch.

## Requirements
- Python 3.x
- Libraries: `numpy`, `torch`, `opencv-python`, `mediapipe`, `Pillow`,`pickle`
  
Install the required dependencies used in our work with cmd:
```bash
pip install numpy torch opencv-python mediapipe pillow

## Project Structure
DanceDemo.py: Main script for running the demo to see the execution of different methods.

GenNearest.py: Generates images by finding the closest skeleton match between source and target frames using Euclidean distance.

GenVanillaNN.py: Implements a neural network that directly trains a model with vector skeletons as input and target images as output, with the objective to generate images from unseen skeleton vectors.

GenVanillaNN2.py: Implements a neural network that directly trains a model with skeleton images as input and target images as output, with the objective to generate images from unseen skeleton images.

GenGAN.py: Implements a neural network that directly trains a model with skeleton images as input and target images as output, adding a discriminator for improved image realism that checks if the image is fake or real and tries to improve its realism until it predicts a real image, with the objective to generate realistic images from unseen skeleton images.

Skeleton.py: Contains functions for processing skeleton data. It detects and visualizes human skeletal landmarks on an image, allowing for full or reduced skeleton displays and joint connections.

VideoSkeleton.py: Processes a video to detect and crop skeletons in frames, saves the results, and visualizes each frame with overlaid skeletons, enabling analysis of skeletal data in videos.

VideoReader.py: Reads, manages, and processes frames from a video file, allowing access to frame dimensions, total frames, FPS, and seamless frame retrieval for display or further processing.

## Data Preparation
To convert a video into frames and extract skeleton data, run:
python VideoSkeleton.py --input your_video.mp4 --output data/

## Models
##1. Nearest Neighbor Model (GenNearest)
Generates an image from a given skeleton posture(source) by finding the closest matching skeleton in a target video using Euclidean distance. It selects the corresponding image from the video(Target) that has the most similar skeleton to the input one(source).
##2. Direct Neural Network Model (GenVanillaNN)
## Neural Network Model = Input(Skeleton Vector) ==>  Output (Genrated Target Image)
This neural network directly trains a model with vector skeletons as input and target images as output, aiming to generate images from unseen skeleton vectors. Unlike the nearest neighbor method, which finds the closest skeleton in the target set using Euclidean distance (often resulting in a slightly different pose due to distance limitations).
This neural network can generate new skeleton poses by learning the mapping from skeleton coordinates to target images. After training, it enables the generation of images based on novel skeleton vectors rather than limited pre-existing matches.

Training: Initialize and train with:
gen = GenVanillaNN(videoSke)
gen.train(n_epochs=20)

## Generating Images: 
Produce an image from a new skeleton vector input with:
image = gen.generate(skeleton_input)





##3. Alternative Neural Network Model (GenVanillaNN2)
## Neural Network Model = Input(Skeleton Image) ==>  Output (Genrated Target Image)
This neural network directly trains a model with skeleton images as input and target images as output, aiming to generate images from unseen skeleton poses. Unlike the nearest neighbor method, which finds the closest skeleton in the target set using Euclidean distance (often resulting in a slightly different pose due to distance limitations).
This neural network can generate new skeleton poses by learning the mapping from skeleton images to target images. After training, it enables the generation of images based on novel skeleton images rather than being restricted to pre-existing matches.
Training: Initialize and train with:
gen2 = GenVanillaNN2(videoSke)
gen2.train(n_epochs=20)

## Generating Images: Produce an image from skeleton data with:
image = gen2.generate(skeleton_input)




##4. GAN Model (GenGAN)
## GAN Model = Input (Skeleton Image) ==> Output (Generated Target Image with Discriminator)
Uses a GAN framework with a discriminator for enhanced image quality

The GAN model, like the neural network, directly trains on skeleton images as input and target images as output, aiming to generate realistic images from new skeleton poses. However, unlike the basic neural network, the GAN model includes an additional discriminator component. The discriminator's role is to distinguish between real and generated images, pushing the generator to improve the realism of its output. This adversarial setup helps the GAN model not only generate novel skeleton poses but also enhance image quality, achieving results that appear more authentic by iteratively refining until the generated images look indistinguishable from real ones.




##Training the Models
All models can be trained with train() functions, optimizing with MSE or GAN losses depending on the architecture.
The default training duration is set to 1000 epochs, ensuring sufficient iterations for the model to learn complex image mappings from skeleton inputs effectively.

## Usage Instructions
Running the Code with a Trained Network :
After training and saving each model, use `DanceDemo.py` to animate a target person's movements based on skeleton poses extracted from a source video. This script takes the skeleton data saved using `VideoSkeleton.py`, which prepares the frames by detecting, cropping, and overlaying skeleton poses on the target frames. Make sure to have the processed skeleton data ready before running the animation demo.

### Training the Networks
To train the models, use the `train()` method on the desired model class (`GenNearest`, `GenVanillaNN`, `GenVanillaNN2`, or `GenGAN`). Each model class has its own approach to synthesizing images, with different structures and loss functions. While `GenNearest` uses a baseline matching technique, `GenVanillaNN` and `GenVanillaNN2` are neural networks trained with Mean Squared Error (MSE) loss, and `GenGAN` uses a GAN framework that incorporates an adversarial loss to enhance realism. Set the `n_epochs` parameter as required, with 1000 epochs as a default for stability and quality.


## Explanation
This project explores various image synthesis techniques for generating realistic images based on human posture. The Nearest Neighbor Matching approach serves as a baseline, where the closest skeleton matches are selected using Euclidean distance, aligning source skeleton poses with the most similar target poses. Direct Neural Networks (GenVanillaNN and GenVanillaNN2) are trained to map skeleton data (either as vectors or images) to corresponding target images, using simple neural architectures. Each network is configured differently, with GenVanillaNN working with skeleton vectors as input and GenVanillaNN2 processing skeleton images. Lastly, the GAN-based Approach (GenGAN) integrates a discriminator to learn adversarially, improving image realism by generating more convincing and high-quality outputs. This method outperforms the simpler models by producing images with better details and natural-looking poses. Collectively, these techniques allow for realistic animation transfer, enabling the generation of target person movements based on the skeleton data from a source video.