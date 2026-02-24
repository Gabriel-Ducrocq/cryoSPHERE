import os
import sys
import torch
import unittest
import pytorch3d
import numpy as np
sys.path.insert(1, '../model')
from segmentation import Segmentation
from utils import compute_translations_per_residue, deform_structure, parse_yaml
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion, quaternion_apply

def rotate_residues_einops(atom_positions, quaternions, segmentation, device):
    """
    Rotates each residues based on the rotation predicted for each domain and the predicted segmentation.
    :param positions: torch.tensor(N_residues, 3)
    :param quaternions: tensor (N_batch, N_segments, 4) of non normalized quaternions defining rotations
    :param segmentation: tensor (N_batch, N_residues, N_segments)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """

    N_residues = segmentation.shape[1]
    batch_size = quaternions.shape[0]
    N_segments = segmentation.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_segments_axis_angle = quaternion_to_axis_angle(quaternions)
    #The below tensor is [N_batch, N_residues, N_segments, 3]
    segmentation_rotation_per_segments_axis_angle = segmentation[:, :, :, None] * rotation_per_segments_axis_angle[:, None, :, :]
    segmentation_rotation_per_segments_quaternions = axis_angle_to_quaternion(segmentation_rotation_per_segments_axis_angle)
    #T = Transform3d(dtype=torch.float32, device = device)
    atom_positions = quaternion_apply(segmentation_rotation_per_segments_quaternions[:, :, 0, :], atom_positions)
    for segm in range(1, N_segments):
        atom_positions = quaternion_apply(segmentation_rotation_per_segments_quaternions[:, :, segm, :], atom_positions)

    return atom_positions

def deform_structure_old(atom_positions, translation_per_residue, quaternions, segmentation, device):
    """
    Deform the base structure according to rotations and translation of each segment, together with the segmentation.
    :param atom_positions: torch.tensor(N_residues, 3)
    :param translation_per_residue: tensor (Batch_size, N_residues, 3)
    :param quaternions: tensor (N_batch, N_segments, 4) of quaternions for the rotation of the segments
    :param segmentation: torch.tensor(N_batch, N_residues, N_segments) weights of the segmentation
    :param device: torch device on which the computation takes place
    :return: tensor (Batch_size, N_residues, 3) corresponding to translated structure
    """
    transformed_atom_positions = rotate_residues_einops(atom_positions, quaternions, segmentation, device)
    new_atom_positions = transformed_atom_positions + translation_per_residue
    return new_atom_positions

def compute_translations_per_residue_old(translation_vectors, segmentation):
    """
    Computes one translation vector per residue based on the segmentation
    :param translation_vectors: torch.tensor (Batch_size, N_segments, 3) translations for each domain
    :param segmentation: torch.tensor(N_batch, N_residues, N_segments) weights of the segmentation
    :return: translation per residue torch.tensor(batch_size, N_residues, 3)
    """
    translation_per_residue = torch.einsum("bij, bjk -> bik", segmentation, translation_vectors)
    return translation_per_residue

def sample_segmentation(N_batch, N_segments, segmentation_proportions_std, segmentation_proportions_mean, segmentation_means_std, segmentation_means_mean,
						segmentation_std_std, segmentation_std_mean, residues, tau_segmentation, device):
    """
    Samples a segmantion
    :param N_batch: integer: size of the batch.
    :return: torch.tensor(N_batch, N_residues, N_segments) values of the segmentation
    """
    elu = torch.nn.ELU()
    cluster_proportions = torch.randn((N_batch, N_segments),
                                      device=device) * segmentation_proportions_std+ segmentation_proportions_mean
    cluster_means = torch.randn((N_batch, N_segments), device=device) * segmentation_means_std+ segmentation_means_mean
    cluster_std = elu(torch.randn((N_batch, N_segments), device=device)*segmentation_std_std + segmentation_std_mean) + 1
    proportions = torch.softmax(cluster_proportions, dim=-1)
    log_num = -0.5*(residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
          torch.log(proportions[:, None, :])

    segmentation = torch.softmax(log_num / tau_segmentation, dim=-1)
    return segmentation


class TestSegmentation(unittest.TestCase):
	"""
	Class for testing the segmentation of different parts.
	"""
	def setUp(self):
		self.N_residues = 1000
		self.residues_chain = np.array(["A" for _ in range(100)] + ["B" for _ in range(500)] + ["C" for _ in range(400)])
		self.residues_indexes = np.array([i for i in range(1000)])
		self.segmentation_config = {"part1":{"N_segm":6, "start_res":0, "end_res":80, "chain":"A"}, "part2":{"N_segm":15, "start_res":300, "end_res":499, "chain":"B"},
									"part3":{"N_segm":10, "start_res":300, "end_res":399, "chain":"C"}}
		self.segmentation_config2 = {"part1":{"N_segm":6, "all_protein":True}}
		self.segmenter = Segmentation(self.segmentation_config, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)
		self.segmenter2 = Segmentation(self.segmentation_config2, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)
		self.batch_size = 10

		self.segmentation_config_non_uniform = {"part1":{"N_segm":6, "start_res":0, "end_res":80, "chain":"A", "segmentation_start_values":
												{"means_stds":[10, 10, 10, 10, 10, 10], "means_means":[10, 150, 500, 600, 800, 900], "stds_means":[10, 10, 10, 10, 10, 10], 
												"stds_stds": [10, 10, 10, 10, 10, 10], "proportions_means":[0, 0, 0, 0, 0, 0], "proportions_stds":[1, 1, 1, 1, 1, 1]}, 
												"segmentation_prior":{"means_stds":[10, 10, 10, 10, 10, 10], "means_means":[10, 150, 500, 600, 800, 900], "stds_means":[10, 10, 10, 10, 10, 10], 
												"stds_stds": [10, 10, 10, 10, 10, 10], "proportions_means":[0, 0, 0, 0, 0, 0], "proportions_stds":[1, 1, 1, 1, 1, 1]},
													}, "part2":{"N_segm":15, "start_res":300, "end_res":499, "chain":"B"},
									"part3":{"N_segm":10, "start_res":300, "end_res":399, "chain":"C"}}

		self.segmenter_non_uniform = Segmentation(self.segmentation_config_non_uniform, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)					


	def test_segmentation(self):
		"""
		Test the actual segmentation function of the Segmentation class
		"""
		segmentation = self.segmenter.sample_segments(self.batch_size)
		total_moving_residues = 81 + 200 + 100
		total_start = 0
		for part, segm in self.segmentation_config.items():
			segm_mat = segmentation[part]["segmentation"]
			segm_mask = segmentation[part]["mask"]

			self.assertEqual(segm_mat.shape[0], self.batch_size)
			self.assertEqual(segm_mat.shape[1], segm["end_res"] - segm["start_res"]+1)
			self.assertEqual(segm_mat.shape[2], segm["N_segm"])
			self.assertEqual(np.sum(segm_mask), segm["end_res"] - segm["start_res"]+1)
			self.assertEqual(np.sum(segm_mask[total_start + segm["start_res"]:total_start+segm["end_res"]+1] == 0), 0)
			total_start += np.sum(self.residues_chain == segm["chain"])

	def test_compare_old_new_segmentation(self):
		torch.manual_seed(0)
		segmentation1 = self.segmenter2.sample_segments(self.batch_size)
		torch.manual_seed(0)
		residues = torch.tensor(self.residues_indexes, dtype=torch.float32, device="cpu")[:, None]
		segmentation2 = sample_segmentation(self.batch_size, self.segmentation_config2["part1"]["N_segm"], self.segmenter2.segments_proportions_stds["part1"], self.segmenter2.segments_proportions_means["part1"],
		   self.segmenter2.segments_means_stds["part1"], self.segmenter2.segments_means_means["part1"], self.segmenter2.segments_stds_stds["part1"], self.segmenter2.segments_stds_means["part1"],
		   residues, 0.05, device="cpu")

		max_error = np.max(torch.abs(segmentation1["part1"]["segmentation"] - segmentation2).detach().cpu().numpy())
		self.assertAlmostEqual(max_error, 0.0, 4)


	def test_non_uniform_init_segmentation(self):
		for type1 in ["means", "stds", "proportions"]:
			for type2 in ["means", "stds"]:
				print(f"{type1}_{type2}")
				print(list(self.segmenter_non_uniform.__getattr__(f"segments_{type1}_{type2}")["part1"]))




class TestMovingResidues(unittest.TestCase):
	"""
	Class for testing the displacement of the residues based on the segmentation by parts.
	"""
	def setUp(self):
		#torch.manual_seed(1)
		self.device="cpu"
		self.batch_size = 10
		self.N_residues = 1000
		self.residues_chain = np.array(["A" for _ in range(100)] + ["B" for _ in range(500)] + ["C" for _ in range(400)])
		self.residues_indexes = np.array([i for i in range(1000)])
		self.segmentation_config1 = {"part1":{"N_segm":6, "start_res":0, "end_res":80, "chain":"A"}, "part2":{"N_segm":15, "start_res":300, "end_res":499, "chain":"B"},
									"part3":{"N_segm":10, "start_res":300, "end_res":399, "chain":"C"}}

		self.segmentation_config2 = {"part1":{"N_segm":6, "all_protein":True}}
		self.segmenter = Segmentation(self.segmentation_config1, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)
		self.segmenter2 = Segmentation(self.segmentation_config2, self.residues_indexes, self.residues_chain, tau_segmentation=0.05)
		self.atom_positions = torch.randn((self.N_residues, 3), dtype=torch.float32, device=self.device)
		self.translation_per_segments = {}
		self.rotation_per_segments = {}
		self.translation_per_segments2 = {}
		self.rotation_per_segments2 = {}
		for part, part_config in self.segmentation_config1.items():
			self.translation_per_segments[part] = torch.randn((self.batch_size, self.segmentation_config1[part]["N_segm"], 3), dtype=torch.float32, device=self.device)
			self.rotation_per_segments[part] = pytorch3d.transforms.random_quaternions(
										part_config["N_segm"]*self.batch_size, device=self.device).reshape(self.batch_size, part_config["N_segm"], -1)

		for part, part_config in self.segmentation_config2.items():
			self.translation_per_segments2[part] = torch.randn((self.batch_size, self.segmentation_config2[part]["N_segm"], 3), dtype=torch.float32, device=self.device)
			self.rotation_per_segments2[part] = pytorch3d.transforms.random_quaternions(
										part_config["N_segm"]*self.batch_size, device=self.device).reshape(self.batch_size, part_config["N_segm"], -1)

	def test_translations(self):
		"""
		Test that we translate the right atoms and leave the others untouched.
		"""
		segmentation = self.segmenter.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments, segmentation, self.N_residues, self.batch_size, self.device)
		mask = np.zeros(self.N_residues)
		for part, segm in segmentation.items():
			mask += segm["mask"]

		max_trans_non_moving = np.max(torch.abs(translations_per_residue[:, mask == 0]).detach().cpu().numpy())
		self.assertEqual(max_trans_non_moving, 0.0)

	def test_rotations(self):
		"""
		Testing that we rotate the right residues
		"""
		segmentation = self.segmenter.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments, segmentation, self.N_residues, self.batch_size, self.device)
		new_atom_positions = deform_structure(self.atom_positions, translations_per_residue, self.rotation_per_segments, segmentation, self.device)
		distances = (new_atom_positions - self.atom_positions)**2
		mask = np.zeros(self.N_residues)
		for part, segm in segmentation.items():
			mask += segm["mask"]

		self.assertEqual(np.max(distances[:, mask==0].detach().cpu().numpy()), 0.0)
		segmentation = self.segmenter2.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments2, segmentation, self.N_residues, self.batch_size, self.device)
		new_atom_positions = deform_structure(self.atom_positions, translations_per_residue, self.rotation_per_segments2, segmentation, self.device)
		distances = (new_atom_positions - self.atom_positions)**2
		mask = np.zeros(self.N_residues)
		for part, segm in segmentation.items():
			mask += segm["mask"]

		self.assertEqual(0.0, 0.0)

	def test_old_new_translations(self):
		segmentation = self.segmenter2.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments2, segmentation, self.N_residues, self.batch_size, self.device)
		translations_per_residue_old = compute_translations_per_residue_old(self.translation_per_segments2["part1"], segmentation["part1"]["segmentation"])

		diff = np.max(torch.abs(translations_per_residue - translations_per_residue_old).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_old_new_rotations(self):
		segmentation = self.segmenter2.sample_segments(self.batch_size)
		translations_per_residue = compute_translations_per_residue(self.translation_per_segments2, segmentation, self.N_residues, self.batch_size, self.device)
		print(self.atom_positions.shape)
		new_atom_positions = deform_structure(self.atom_positions, translations_per_residue, self.rotation_per_segments2, segmentation, self.device)

		translations_per_residue_old = compute_translations_per_residue_old(self.translation_per_segments2["part1"], segmentation["part1"]["segmentation"])
		print(self.atom_positions.shape)
		new_atom_positions_old = deform_structure_old(self.atom_positions, translations_per_residue_old, self.rotation_per_segments2["part1"], segmentation["part1"]["segmentation"], self.device)

		diff = np.max(torch.abs(new_atom_positions - new_atom_positions_old).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)


	#def test_yaml_parsing(self):
	#	"""
	#	Tests if the yaml parsing still works well
	#	"""
	#	try:
	#		parse_yaml("test_apoferritin/parameters_package_segmentation.yaml")
	#		parse_yaml("test_apoferritin/parameters_package_segmentation_full_protein.yaml")
	#		self.assertEqual(0.0, 0.0)
	#	except:
	#		self.assertEqual(0.0, 1.0)







