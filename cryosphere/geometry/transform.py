"""
This file contains all the functions for applying SE(3) transforms to a set of points.
There are two main components: apply_rotation and apply_translation functions.

The apply_rotation function applies a rotation to a set of points in 3D. It assumes that the rotation is expressed as a rotation matrix in the left multiplication
convention. The matrix has counterclockwise convention.

The apply_translation function applies a translation to set of points in 3D. 

Both function support batching: a single set of points can be rotated/translated by multiple rotations/translations, resulting in multiple set of points, each of which has been
rotated/translated according to only one rotation/translation.

NB: The rotation/translation tensor are assumed to have a batch dimension, even if it is 1.
"""
import torch


def apply_rotation(points: torch.tensor, rotation_matrix: torch.tensor) -> torch.tensor:
	"""
	This function applies one or several rotation matrices to the set of set of points in 3D.
	The rotation matrix follows the left multiplication convention and expresses the rotation in the counter clockwise convention.
	The rotation_matrix tensor is supposed to have the first component being the batch component.

	:param points: tensor of points in 3D, of shape (batch_size, N_points, 3).
	:param rotation_matrix: set of rotation matrices, of shape (batch_size, 3, 3)
	:return: a set of multiple set of points, of shape (batch_size, 3), where each batch component is the set of point rotated by the corresponding component in the
		rotation_matrix tensor.
	"""
	rotated_points = torch.einsum("bnij, bnj -> bni", rotation_matrix, points)
	return rotated_points

def apply_translation(points:torch.tensor, translation_vector: torch.tensor) -> torch.tensor:
	"""
	This function applies translation vectors to each points in the tensor.
	The translation_vector tensor is supposed to have the first component being the batch component, even if it has dimension 1.

	:param points: tensor of points in 3D, of shape (batch_size, N_points, 3).
	:param translation_vector: set of translation vectors, of shape (batch_size, N_points, 3)
	:return: a set of multiple set of points, of shape (batch_size, N_points, 3), where each batch component is the set of point translated by the corresponding component in the
		translation_vectors tensor.
	"""
	translated_points = points + translation_vector
	return translated_points




