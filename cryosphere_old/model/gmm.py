import torch
import numpy as np
from typing import Union
from dataclasses import dataclass



@dataclass
class Gaussian:
    mus: Union[torch.Tensor, np.ndarray]
    sigmas: Union[torch.Tensor, np.ndarray]
    amplitudes: Union[torch.Tensor, np.ndarray]



class BaseGrid(torch.nn.Module):
	"""
	Grid spanning origin, to origin + (side_shape - 1) * voxel_size, for the coordinate of each pixel.
	"""
	def __init__(self, side_n_pixels, voxel_size, origin=None, device="cpu"):
		"""
		:param side_n_pixels: integer, number of pixel on each side of the image.
		:param voxel_size: float, size of each pixel in Å.
		:param origin: float, origin on the image.
		:param device: torch device on which to perform the computations.
    	"""
		super().__init__()
		self.side_n_pixels = side_n_pixels
		self.voxel_size = voxel_size
		if not origin:
			origin = 0

		self.origin = origin

		line_coords = torch.linspace(origin, (side_n_pixels - 1) * voxel_size + origin, side_n_pixels, device=device)
		self.register_buffer("line_coords", line_coords)
		[xx, yy] = torch.meshgrid([self.line_coords, self.line_coords], indexing="ij")
		plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
		self.register_buffer("plane_coords", plane_coords)
		self.plane_shape = (self.side_n_pixels, self.side_n_pixels)

		[xx, yy, zz] = torch.meshgrid([self.line_coords, self.line_coords, self.line_coords])
		vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
		self.register_buffer("vol_coords", vol_coords)
		self.vol_shape = (self.side_n_pixels, self.side_n_pixels, self.side_n_pixels)



class EMAN2Grid(BaseGrid):
	"""EMAN2 style grid for the coordinates of each pixel.
	origin set to -(side_shape // 2) * voxel_size
	"""
	def __init__(self, side_shape, voxel_size, device="cpu"):
		"""
		:param side_shape: integer, number of pixel on each side of the image.
		:param voxel_size: float, size of each pixel in Å.
		"""
		origin = -side_shape // 2 * voxel_size
		super().__init__(side_n_pixels=int(side_shape), voxel_size=voxel_size, origin=origin, device=device)




