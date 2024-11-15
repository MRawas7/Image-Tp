import numpy as np
import cv2
import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim

from VideoSkeleton import VideoSkeleton  # Ensure this is correctly implemented
from Skeleton import Skeleton  # Ensure this is correctly implemented

torch.set_default_dtype(torch.float32)

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
        ske = ske.to(torch.float32).reshape(-1, 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()  # Ensure it's on CPU
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = (numpy_image + 1) / 2 * 255  # Scale back to [0, 255]
        numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)  # Ensure proper uint8 format
        return numpy_image


class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(26, 128, kernel_size=4, stride=1, padding=0),  # (26, 1, 1) -> (128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (128, 4, 4) -> (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (64, 8, 8) -> (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # (32, 16, 16) -> (3, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),      # (3, 32, 32) -> (3, 64, 64)
            nn.Tanh()  # Ensure output is in the range [-1, 1] for better stability
        )

    def forward(self, z):
        z = z.view(z.size(0), 26, 1, 1)  # Reshape to (batch_size, 26, 1, 1)
        img = self.model(z)
        return img


class GenVanillaNN():
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.netG = GenNNSkeToImage()
        self.filename = 'data/DanceGenVanillaFromSke.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            self.netG.load_state_dict(torch.load(self.filename))

    def train(self, n_epochs=20):
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Use Mean Squared Error loss for image generation
        self.netG.train()

        for epoch in range(n_epochs):
            for ske, img in self.dataloader:
                optimizer.zero_grad()
                output = self.netG(ske)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

        torch.save(self.netG.state_dict(), self.filename)  # Save the model after training


        print('Finished Training')

    def generate(self, ske):
        """ Generator of image from skeleton """
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)  # make a batch
        normalized_output = self.netG(ske_t_batch)
        res = self.dataset.tensor2image(normalized_output[0])  # get image 0 from the batch
        return res


if __name__ == '__main__':
    force = False
    n_epoch = 2000  # Set the number of epochs for training
    train = True  # Set to True to train the model

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)  # load from file

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
        if key == 27:  # Esc key to exit
            break

    cv2.destroyAllWindows()  # Ensure all windows are closed after displaying images
