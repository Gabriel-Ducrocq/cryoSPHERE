import torch
import unittest
import sys
sys.path.insert(1, '../model')
from utils import compute_cross_prod_mat, compute_rotation_matrix_in_0


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



