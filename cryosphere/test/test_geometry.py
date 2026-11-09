import torch
import unittest
import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
sys.path.insert(1, '../model')
from utils import compute_cross_prod_mat, compute_rotation_matrix_in_0, remove_global_translation


class TestComputeRotationMatInZero:
    """
    Testing the set of functions used for computing the gradient of the loss w.r.t to the axis-angel poses.
    """
    def test_compute_cross_prod_mat(self):
        w = torch.zeros(10, 3)
        skew_mat = compute_cross_prod_mat(w)
        assert torch.equal(skew_mat, torch.zeros_like(skew_mat)), f"""The skew symmetric matrix corresponding to a zero vector should be
                                                    the 0 matrix. Currently is {skew_mat}."""

        w = torch.ones(1, 3)
        w[:, 0] = 1
        w[:, 1] = 2
        w[:, 2] = 3
        correct_skew_sym_mat = torch.zeros(1, 3, 3)
        correct_skew_sym_mat[:, 0, 1] = -w[:, -1]
        correct_skew_sym_mat[:, 1, 0] = w[:, -1]
        correct_skew_sym_mat[:, 0, 2] = w[:, 1]
        correct_skew_sym_mat[:, 2, 0] = -w[:, 1]
        correct_skew_sym_mat[:, 1, -1] = -w[:, 0]
        correct_skew_sym_mat[:, -1, 1] = w[:, 0]

        skew_mat = compute_cross_prod_mat(w)
        assert torch.equal(skew_mat, correct_skew_sym_mat), f"""Recovered skew symmetric matrix does not match 
                                                                expectations. Currently {skew_mat} while expected {correct_skew_sym_mat}."""


    def compute_rotation_matrix_in_zero(self):
        w = torch.zeros(34, 3)
        computed_rot_mat = compute_rotation_matrix_in_0(w)
        expect_rot_mat = torch.eye(3, 3)[None].repeat(34, 1, 1)
        assert torch.equal(computed_rot_mat, expect_rot_mat), f"""Recovered rotation matrix does not match
                                                                  the identity matrix {computed_rot_mat}."""



class TestRemoveGlobalTranslation:
    """
    Testing that the average removal of translation works well
    """
    def setup_method(self):
        torch.manual_seed(123)
        self.x = torch.randn((128, 6, 3))
        self.x_base = torch.randn((6, 3))

    def test_centroid_matches_base(self):
        x_centered = remove_global_translation(self.x, self.x_base)
        x_centered_averaged = torch.mean(x_centered, dim=1)
        expected_avg = torch.mean(self.x_base, dim=0)
        assert torch.all(torch.isclose(x_centered_averaged, expected_avg[None,:], atol=1e-7)), f"""The average of the centered translation should be
                                                                    {expected_avg}, currently {x_centered_averaged}."""

    def test_shape_preserved(self):
        """
        The pairwise distances should not be changed
        """
        out = remove_global_translation(self.x, self.x_base)
        torch.testing.assert_close(out[:, :, None] - out[:, None], self.x[:, :, None] - self.x[:, None])

    def test_invariance_global_translation(self):
        """
        The results should not depend on a global translation
        """
        global_trans = torch.randn((128, 1, 3))
        out1 = remove_global_translation(self.x + global_trans, self.x_base)
        out2 = remove_global_translation(self.x, self.x_base)
        torch.testing.assert_close(out1, out2)


    def test_no_gradient_along_global_translation(self):
        """
        Verifies there is no gradient in direction of global shifting. This is the property that we care the most
        about as this prevents the network from learning a global shift
        """
        x = self.x.clone().requires_grad_()
        out = remove_global_translation(x, self.x_base)
        (out*torch.randn_like(out)).sum().backward()
        #We take the sum of the gradients along the residues. This way, we see if the gradient has a non-null average
        #component that will shift the entire structure.
        torch.testing.assert_close(x.grad.sum(1), torch.zeros((128, 3)), atol=1e-5, rtol=0)
