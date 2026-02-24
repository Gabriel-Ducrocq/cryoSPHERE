import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, device, network_type="decoder"):
        """
        Multi layer perceptron class. If it is of encoder type, we split the output in half along the last dimension, and pass the second half to an ELU + 1 layer because 
        we want a std for the latent distribution > 0.
        :param in_dim: integer, input dimension
        :param out_dim: integer, output dimension
        :param intermediate_dim: list of integer, size of the intermediate dimensions
        :param device: torch device on which we perform the computations
        :param network_type: str, "encoder" of "decoder".
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.type=network_type
        self.out_dim = out_dim
        self.output_ELU = torch.nn.ELU()
        assert type(intermediate_dim) == type([]), "intermediate_dim should be a list containing the size of the intermediate layers."
        self.num_hidden_layers = len(intermediate_dim)
        self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim[0], device=device), nn.LeakyReLU())
        self.output_layer = nn.Sequential(nn.Linear(intermediate_dim[-1], out_dim, device=device))
        list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim[i], intermediate_dim[i+1], device=device), nn.LeakyReLU())
                         for i in range(self.num_hidden_layers-1)]
        self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])


    def forward(self, x):
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        output = self.output_layer(hidden)
        if self.type == "encoder":
            latent_mean = output[:, :int(self.out_dim/2)]
            latent_std = self.output_ELU(output[:, int(self.out_dim/2):]) + 1
            return latent_mean, latent_std

        return output