import roma
import torch
from cryosphere.model import utils




class TestResidueRotation:
    torch.manual_seed(0)
    batch_size = 128
    N_atoms = 10
    N_segments = 5
    atom_pos = torch.randn(N_atoms, 3)
    random_rot_mat = roma.random_rotmat((batch_size, N_segments))
    random_r6 = random_rot_mat[:, :, :, :2]
    # Since roma uses the xyzw convention through rotmat_to_unitquat, but my code assumes wxyz convention, I need to change it
    # To that convention firt
    random_quat = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(random_rot_mat))
    segmentation = torch.zeros(batch_size, N_atoms, N_segments)
    indexes = torch.randint(low=0, high=N_segments, size=(batch_size, N_atoms))
    segmentation.scatter_(2, indexes.unsqueeze(-1), 1)
    rotation_per_residue = random_rot_mat[torch.arange(batch_size)[:, None], indexes]
    rotated_atoms = torch.einsum("bijk, ik -> bij", rotation_per_residue, atom_pos)
    def test_r6_rotation(self):
        r_r6 = utils.rotate_residues_einops_r6(self.atom_pos, self.random_r6, self.segmentation)
        assert torch.all(torch.isclose(r_r6, self.rotated_atoms, rtol=1e-5,
                                       atol=1e-6)), """The rotated atoms computed by hand and the r6 rotation function
                                                     do not match."""

    def test_quat_rotation(self):
        r_quat = utils.rotate_residues_einops(self.atom_pos, self.random_quat, self.segmentation)
        assert torch.all(torch.isclose(r_quat, self.rotated_atoms, rtol=1e-5,
                                       atol=1e-6)), """The rotated atoms computed by hand and the quaternion rotation function
                                                     do not match."""

    def test_quat_and_r6_rotation(self):
        r_r6 = utils.rotate_residues_einops_r6(self.atom_pos, self.random_r6, self.segmentation)
        r_quat = utils.rotate_residues_einops(self.atom_pos, self.random_quat, self.segmentation)
        assert torch.all(torch.isclose(r_quat, r_r6, rtol=1e-5, atol=1e-6)), """The rotated atoms computed by the r6 
                                                                                rotation and the quaternion rotation 
                                                                                function do not match."""




