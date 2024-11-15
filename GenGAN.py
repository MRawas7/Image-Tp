
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import *


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Input: (batch_size, 3, 64, 64)
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename, map_location=torch.device('cpu'))

    def train(self, n_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = self.netG.to(device)
        self.netD = self.netD.to(device)

        # Initialize optimizers with adjusted learning rates
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0004, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # Loss functions
        criterion_GAN = nn.BCELoss()
        criterion_pixel = nn.MSELoss()

        # Training noise level
        noise_factor = 0.1

        print(f"Starting training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            total_d_loss = 0
            total_g_loss = 0

            for i, (skeletons, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)

                # Move data to device
                real_images = real_images.to(device)
                skeletons = skeletons.to(device)

                # Add noise to real images for discriminator
                real_images_noisy = real_images + torch.randn_like(real_images) * noise_factor

                # Create labels with smoothing
                real_target = torch.ones(batch_size, device=device, dtype=torch.float32) * 0.9
                fake_target = torch.zeros(batch_size, device=device, dtype=torch.float32) + 0.1

                ############################
                # Train Discriminator
                ############################
                self.netD.zero_grad()

                # Train with real images
                output_real = self.netD(real_images_noisy).view(-1)
                d_loss_real = criterion_GAN(output_real, real_target)

                # Train with fake images
                fake_images = self.netG(skeletons)
                fake_images_noisy = fake_images + torch.randn_like(fake_images) * noise_factor
                output_fake = self.netD(fake_images_noisy.detach()).view(-1)
                d_loss_fake = criterion_GAN(output_fake, fake_target)

                # Combined D loss
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                # Compute gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_images, fake_images.detach())
                d_loss = d_loss + 10.0 * gradient_penalty

                d_loss.backward()
                optimizerD.step()

                ############################
                # Train Generator
                ############################
                if i % 1 == 0:  # Update G less frequently than D
                    self.netG.zero_grad()

                    # GAN loss
                    output_fake = self.netD(fake_images).view(-1)
                    g_loss_GAN = criterion_GAN(output_fake, real_target)

                    # Pixel-wise loss
                    g_loss_pixel = criterion_pixel(fake_images, real_images)

                    # Combined G loss
                    g_loss = g_loss_GAN + 100 * g_loss_pixel
                    g_loss.backward()
                    optimizerG.step()

                # Print statistics
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()

                if i % 10 == 0:
                    print(f'[{epoch + 1}/{n_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f} '
                          f'D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f}')

            # Print epoch statistics
            avg_d_loss = total_d_loss / len(self.dataloader)
            avg_g_loss = total_g_loss / len(self.dataloader)
            print(f'Epoch [{epoch + 1}/{n_epochs}], Average Loss - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}')

            # Save model periodically
            if (epoch + 1) % 5 == 0:
                print(f"Saving model at epoch {epoch + 1}...")
                torch.save(self.netG, self.filename)

        # Save final model
        print("Training completed. Saving final model...")
        torch.save(self.netG, self.filename)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        device = real_samples.device
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Get discriminator output for interpolated images
        d_interpolated = self.netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    def generate(self, ske):
        """Generator of image from skeleton"""
        # Get the device the model is on
        device = next(self.netG.parameters()).device

        # Prepare skeleton input
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten())
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1, Skeleton.reduced_dim, 1, 1)  # Add batch and spatial dimensions

        # Move to same device as model
        ske_t = ske_t.to(device)

        # Set model to evaluation mode
        self.netG.eval()

        with torch.no_grad():  # No need to track gradients during generation
            # Generate image
            normalized_output = self.netG(ske_t)

            # Move output back to CPU for image processing
            normalized_output = normalized_output.cpu()

            # Convert to image format using dataset's tensor2image function
            res = self.dataset.tensor2image(normalized_output[0])

        return res



if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(200) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

