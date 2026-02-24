import os
import sys
path = os.path.abspath("model")
sys.path.append(path)
import torch
import unittest
import sys
sys.path.insert(1, '../model')
import numpy as np
from ctf import CTF
from dataset import ImageDataSet
from utils import parse_yaml
from torch.utils.data import DataLoader
import inspect

class TestCsStarEquivalenceDataset(unittest.TestCase):
	"""
	This class takes as input a cryoSparc dataset and the same dataset converted to star (RELION) format. 
	It tests if the CTF, images and poses are the same
	"""
	def setUp(self):
		self.star_config = {"file": "test_apoferritin/particles/particles.star"}
		self.cs_file = {"file": "test_apoferritin/J25_split_0_exported.cs"}
		print(ImageDataSet.__init__.__code__.co_varnames)
		self.cs_dataset = ImageDataSet(apix=1.428, side_shape=256, star_cs_file_config=self.cs_file,particles_path="test_apoferritin")
		self.star_dataset = ImageDataSet(apix=1.428, side_shape=256, star_cs_file_config=self.star_config,particles_path="test_apoferritin/particles/")

	def test_images(self):
		torch.manual_seed(0)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		torch.manual_seed(0)
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(batch_images_cs - batch_images_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_rotations(self):
		torch.manual_seed(1)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		torch.manual_seed(1)
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(batch_poses_cs - batch_poses_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_translations(self):
		torch.manual_seed(2)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		torch.manual_seed(2)
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(batch_poses_translation_cs - batch_poses_translation_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_fproj(self):
		torch.manual_seed(3)
		data_loader_cs = iter(DataLoader(self.cs_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))
		torch.manual_seed(3)
		data_loader_star = iter(DataLoader(self.star_dataset, batch_size=1000, shuffle=True, num_workers = 4, drop_last=True))

		_, batch_images_cs, batch_poses_cs, batch_poses_translation_cs, fproj_cs = next(data_loader_cs)
		_, batch_images_star, batch_poses_star, batch_poses_translation_star, fproj_star = next(data_loader_star)

		diff = np.max(torch.abs(fproj_cs - fproj_star).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)


class TestCsStarEquivalenceCTF(unittest.TestCase):
	"""
	This class takes as input a cryoSparc dataset and the same dataset converted to star (RELION) format. 
	It tests if the CTF, images and poses are the same
	"""
	def setUp(self):
		self.ctf_star = CTF.create_ctf({"file":"test_apoferritin/particles/particles.star"}, device="cpu", apix = 1.428, side_shape = 256)
		self.ctf_cs = CTF.create_ctf({"file":"test_apoferritin/J25_split_0_exported.cs", "abinit":False, "hetrefine":False}, device="cpu", apix = 1.428, side_shape = 256)

	def test_dfU(self):
		diff = np.max(torch.abs((self.ctf_cs.dfU - self.ctf_star.dfU)/self.ctf_star.dfU).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_dfV(self):
		diff = np.max(torch.abs((self.ctf_cs.dfV - self.ctf_star.dfV)/self.ctf_star.dfV).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_dfang(self):
		diff = np.max(torch.abs((self.ctf_cs.dfang - self.ctf_star.dfang)/self.ctf_star.dfang).detach().cpu().numpy())
		argm = np.argmax(np.max(torch.abs(self.ctf_cs.dfang - self.ctf_star.dfang).detach().cpu().numpy()))
		print("ARG MAX:", argm)
		self.assertAlmostEqual(diff, 0.0, 4)

	def test_volt(self):
		diff = np.max(torch.abs((self.ctf_cs.volt - self.ctf_star.volt)/self.ctf_star.volt).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_cs(self):
		diff = np.max(torch.abs((self.ctf_cs.cs - self.ctf_star.cs)/self.ctf_star.cs).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_w(self):
		diff = np.max(torch.abs((self.ctf_cs.w - self.ctf_star.w)/self.ctf_star.w).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_phaseShift(self):
		diff = np.max(torch.abs((self.ctf_cs.phaseShift - self.ctf_star.phaseShift)).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)

	def test_scalefactor(self):
		diff = np.max(torch.abs((self.ctf_cs.scalefactor - self.ctf_star.scalefactor)/self.ctf_star.scalefactor).detach().cpu().numpy())
		self.assertAlmostEqual(diff, 0.0, 5)	

	def test_bfactor(self):
		diff = np.max(torch.abs((self.ctf_cs.bfactor - self.ctf_star.bfactor)).detach().cpu().numpy())
		print(self.ctf_cs.bfactor)
		print("\n")
		print(self.ctf_star.bfactor)
		self.assertAlmostEqual(diff, 0.0, 5)



class TestYamlParsing(unittest.TestCase):
	"""
	This class tests whether we still parse the yaml successfully or not, with the cryosparc support. 
	"""
	def test_parse_yaml(self):
		print(inspect.getmembers(CTF, predicate=inspect.ismethod))
		try:
			parse_yaml("test_apoferritin/parameters_package_segmentation.yaml")
			self.assertEqual(0.0, 0.0)
		except:
			self.assertEqual(0.0, 1.0)



if __name__ == '__main__':
    unittest.main()



