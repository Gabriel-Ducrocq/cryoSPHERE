import torch
import unittest
import sys
import os
import roma
from cryosphere.model.utils import compute_segment_weights

path = os.path.abspath("model")
sys.path.append(path)
sys.path.insert(1, '../model')
from utils import compute_cross_prod_mat, compute_rotation_matrix_in_0, remove_global_translation, remove_weighted_average_translation, compute_segment_weights, remove_weighted_average_rotation


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


    def test_compute_rotation_matrix_in_zero(self):
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


class TestRemoveAverageTranslation:
    def setup_method(self):
        torch.manual_seed(123)
        self.translation = torch.randn((128, 10, 3))
        self.weights = torch.softmax(torch.randn(128, 10), dim=-1)

    def test_zero_mean(self):
        translation_prime = remove_weighted_average_translation(self.translation, self.weights)
        avg = torch.sum(translation_prime*self.weights[..., None], dim=1)
        torch.testing.assert_close(avg, torch.zeros_like(avg)), f"""Average tensor should be zero, currently {avg}"""

    def test_invariance_global_translation(self):
        global_trans = torch.randn(128, 1, 3)
        prime = remove_weighted_average_translation(global_trans + self.translation, self.weights)
        prime_global = remove_weighted_average_translation(self.translation, self.weights)
        torch.testing.assert_close(prime, prime_global), f"""The average of a centered tensor {prime} should be the same
                                                             as the average of that tensor translated by a global term {prime_global}."""

    def test_no_gradient_along_global_translation(self):
        translation = self.translation.clone().requires_grad_()
        translation_prime = remove_weighted_average_translation(translation, self.weights)
        (translation_prime*torch.randn_like(translation_prime)).sum().backward()
        torch.testing.assert_close(translation.grad.sum(1), torch.zeros(128, 3)), f"""The gradient of a globally translated
                                                                        gradient should be 0 along any global translation
                                                                        direction. Currently {translation.grad.sum(1)}"""


class TestRemoveAverageRotation:
    def test_common_rotation_removed(self):
        torch.manual_seed(0)
        q = roma.random_unitquat((4, 5))[..., [3, 0, 1, 2]]  # WXYZ
        w = torch.softmax(torch.randn(4, 5), -1)
        G = roma.random_unitquat(4)[:, None].expand(4, 5, 4)  # one global rotation per image (XYZW)
        qG = roma.quat_product(G, q[..., [1, 2, 3, 0]])[..., [3, 0, 1, 2]]  # G R_k, back to WXYZ
        R1 = roma.unitquat_to_rotmat(remove_weighted_average_rotation(q, w)[..., [1, 2, 3, 0]])
        R2 = roma.unitquat_to_rotmat(remove_weighted_average_rotation(qG, w)[..., [1, 2, 3, 0]])
        torch.testing.assert_close(R1, R2, atol=1e-5, rtol=1e-5)  # a common rotation has no effect

    def test_average_is_identity(self):
        torch.manual_seed(1)
        q = roma.random_unitquat((4, 5))[..., [3, 0, 1, 2]]
        w = torch.softmax(torch.randn(4, 5), -1)
        R = roma.unitquat_to_rotmat(remove_weighted_average_rotation(q, w)[..., [1, 2, 3, 0]])
        avg = roma.special_procrustes((R * w[..., None, None]).sum(1))
        torch.testing.assert_close(avg, torch.eye(3).expand(4, 3, 3), atol=1e-5, rtol=1e-5)

    def test_unnormalized_input(self):
        q = roma.random_unitquat((2, 3))[..., [3, 0, 1, 2]]
        w = torch.full((2, 3), 1 / 3)
        torch.testing.assert_close(
            roma.unitquat_to_rotmat(remove_weighted_average_rotation(3.7 * q, w)[..., [1, 2, 3, 0]]),
            roma.unitquat_to_rotmat(remove_weighted_average_rotation(q, w)[..., [1, 2, 3, 0]]))


def make_segmentation(B, sizes, seed=0):
    """sizes: {part: (N_part, K_part)} → random soft segmentations whose rows sum to 1."""
    g = torch.Generator().manual_seed(seed)
    return {part: {"segmentation": torch.softmax(torch.randn(B, n, k, generator=g), dim=-1),
                   "mask": torch.ones(n)}
            for part, (n, k) in sizes.items()}

class TestComputeSegmentWeights:
    def test_shape(self):
        seg = make_segmentation(4, {"part1": (10, 3), "part2": (6, 2)})
        assert compute_segment_weights(seg).shape == (4, 5)

    def test_sum_to_one(self):
        seg = make_segmentation(4, {"part1": (10, 3), "part2": (6, 2)})
        torch.testing.assert_close(compute_segment_weights(seg).sum(-1), torch.ones(4))

    def test_hand_computed(self):
        # part1: 2 residues, 2 segments; part2: 1 residue, 1 segment → N = 3
        seg = {"part1": {"segmentation": torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]), "mask": torch.ones(2)},
               "part2": {"segmentation": torch.tensor([[[1.0]]]), "mask": torch.ones(1)}}
        expected = torch.tensor([[1.5, 0.5, 1.0]]) / 3
        torch.testing.assert_close(compute_segment_weights(seg), expected)

    def test_matches_mean_residue_translation(self):
        """
        Checks that computing the per residue translation average matches the average computed usin the weights:
        1/N times (N1 sum_l proportion_l t_t + N2 sum_k proportion_k t_k) where N is the total number of residues
        moved, N1 (resp N2) total number of residues in part1 (resp part2) and proportion_l = 1/N1 sum_i g_il,
        proportion2 = 1/N2 sum_i g_ik and t_l and t_k are the translations for the part 1 and 2 respectively.
        """
        B = 4
        seg = make_segmentation(B, {"part1": (10, 3), "part2": (6, 2)})
        trans = {"part1": torch.randn(B, 3, 3), "part2": torch.randn(B, 2, 3)}
        # explicit per-residue translations, then their mean over all moved residues
        per_residue = torch.cat([torch.einsum("bnk,bkj->bnj", seg[p]["segmentation"], trans[p]) for p in seg], dim=1)
        expected = per_residue.mean(1)
        # same thing from the weights
        w = compute_segment_weights(seg)
        d = torch.cat([trans[p] for p in seg], dim=1)
        torch.testing.assert_close((w[..., None] * d).sum(1), expected)






