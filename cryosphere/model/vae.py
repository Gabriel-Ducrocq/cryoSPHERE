import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, segmentation_config, latent_dim = None, amortized=True, N_images=None):
        """
        VAE class. This defines all the parameters needed and perform the reparametrization trick.
        :param encoder: object of type MLP, with type "encoder"
        :param decoder: object of type MLP, with type "decoder"
        :param device: torch device on which we want to perform the computations.
        :param N_segments: dictionnary, of number of segments per part of the protein we want to segment.
        :param latent_dim: integer, latent dimension
        :param amortized: bool, whether to perform amortized inference or not
        :param N_images: integer, number of images in the dataset
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.latent_dim = latent_dim
        self.N_images = N_images
        self.amortized = amortized
        self.segmentation_config = segmentation_config
        self.N_total_segments = 0
        for part, part_config in self.segmentation_config.items():
            self.N_total_segments += part_config["N_segm"]

        if not amortized:
            assert N_images, "If using a non amortized version of the code, the number of images must be specified"
            means = torch.randn(N_images, self.latent_dim, dtype=torch.float32, device=device)
            self.latent_variables_mean = torch.nn.Parameter(means, requires_grad=True)
            self.latent_variables_std = torch.nn.Parameter(torch.ones(N_images, self.latent_dim, dtype=torch.float32, device=device), requires_grad=False)


    def sample_latent(self, images, indexes=None):
        """
        Samples latent variables given an image or given an image index if non amortized inference is performed. Apply the reparameterization trick
        :param images: torch.tensor(N_batch, N_pix**2) of input images
        :param indexes: torch.tensor(N_batch, dtype=torch.int) the indexes of images in the batch
        :return: torch.tensor(N_batch, latent_dim) sampled latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std

        """
        if not self.amortized:
            assert indexes is not None, "If using a non-amortized version of the code, the indexes of the images must be provided"
            latent_variables = torch.randn_like(self.latent_variables_mean[indexes, :], dtype=torch.float32, device=self.device)*self.latent_variables_std[indexes, :] + self.latent_variables_mean[indexes, :]
            return latent_variables, self.latent_variables_mean[indexes, :], self.latent_variables_std[indexes, :] 
        else:
            latent_mean, latent_std = self.encoder(images)
            latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                                *latent_std

            return latent_variables, latent_mean, latent_std


    def decode(self, latent_variables):
        """
        Decode the latent variables into a rigid body transformation (rotation and translation) per segment.
        :param latent_variables: torch.tensor(N_batch, latent_dim)
        :return: torch.tensor(N_batch, N_segments, 4) quaternions, torch.tensor(N_batch, N_segments, 3) translations.
        """
        N_batch = latent_variables.shape[0]
        transformations = self.decoder(latent_variables)
        transformations_per_segments = torch.reshape(transformations, (N_batch, self.N_total_segments, 6))
        ones = torch.ones(size=(N_batch, transformations_per_segments.shape[1], 1), device=self.device)
        quaternions_per_segments_all_parts = torch.concat([ones, transformations_per_segments[:, :, 3:]], dim=-1)
        translations_per_segments_all_parts = transformations_per_segments[:, :, :3]
        translations_per_segments = {}
        quaternions_per_segments = {}
        start = 0
        for part, part_config in self.segmentation_config.items():
            n_segments = part_config["N_segm"]
            translations_per_segments[part] = translations_per_segments_all_parts[:, start:(start+n_segments), :]
            quaternions_per_segments[part] = quaternions_per_segments_all_parts[:, start:(start+n_segments)]
            start += n_segments

        return quaternions_per_segments, translations_per_segments



