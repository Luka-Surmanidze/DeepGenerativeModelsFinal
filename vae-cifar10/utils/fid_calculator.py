import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm


class FIDCalculator:
    """
    Calculate Fréchet Inception Distance (FID) from scratch.
    Based on: "GANs Trained by a Two Time-Scale Update Rule Converge to a
               Local Nash Equilibrium" (Heusel et al., 2017)

    FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    where mu1, C1 are mean and covariance of real features
          mu2, C2 are mean and covariance of generated features
    """

    def __init__(self, device):
        self.device = device
        # Load pretrained InceptionV3 model
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

        # Remove final classification layer to get 2048-dim features
        self.inception.fc = nn.Identity()

    def get_activations(self, images):
        """
        Extract InceptionV3 features for a batch of images
        images: tensor of shape [N, 3, H, W] in range [0, 1]
        """
        # Inception expects 299x299 images
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] as expected by Inception
        images = 2 * images - 1

        with torch.no_grad():
            features = self.inception(images)

        return features.cpu().numpy()

    def calculate_activation_statistics(self, images, batch_size=50):
        """
        Calculate mean and covariance of InceptionV3 features
        images: list of image tensors
        """
        activations = []

        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))

            batch = images[start_idx:end_idx]
            if isinstance(batch, list):
                batch = torch.stack(batch)

            batch = batch.to(self.device)
            act = self.get_activations(batch)
            activations.append(act)

        activations = np.concatenate(activations, axis=0)

        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)

        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Fréchet distance between two Gaussians
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return fid
