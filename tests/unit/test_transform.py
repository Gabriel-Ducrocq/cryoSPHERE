import roma
import torch
from cryosphere.geometry.transform import apply_rotation, apply_translation


class TestApplyRotation():
    def test_apply_rotation_output_shape(self):
        batch_size = 128
        n_points = 1006
        points = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = roma.random_rotmat(batch_size*n_points).reshape(batch_size, n_points, 3, 3)
        rotated_points = apply_rotation(points, rotation_matrices)

        assert rotated_points.shape == (batch_size, n_points, 3), f"""The function apply_rotation return an output with wrong
                                        dimensions. Expected {(batch_size, n_points, 3)}, return {rotated_points.shape}."""

    def test_apply_rotation_identity(self):
        batch_size = 128
        n_points = 1006
        points = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = torch.eye(3)[None].repeat((batch_size, 1, 1))
        rotated_points = apply_rotation(points, rotation_matrices)
        assert torch.all(rotated_points == points), """For identity rotations, the points and rotated points should be the 
                                                        same."""
    def test_apply_rotation_preserves_norm(self):
        batch_size = 128
        n_points = 1006
        points = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = roma.random_rotmat(batch_size)
        rotated_points = apply_rotation(points, rotation_matrices)
        norm_points = torch.sum(points**2, dim=-1)
        norm_rotated_points = torch.sum(rotated_points**2, dim=-1)
        assert torch.all(torch.isclose(norm_points ,  norm_rotated_points, rtol=1e-5, atol=1e-8)), """For any set of rotation matrices, the norm of the points
                                                                 and rotated points should be the same."""

    def test_apply_rotation_half_pi(self):
        batch_size = 1
        n_points = 1006
        points = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = torch.zeros((3, 3))[None]
        rotation_matrices[0, 0, 1] = -1
        rotation_matrices[0, 1, 0] = 1
        rotation_matrices[0, 2, 2] = 1
        rotated_points = apply_rotation(points, rotation_matrices)
        expected = points[:,  :, [1, 0 , 2]]
        expected[:, :, 0] *= -1
        assert torch.all(torch.isclose(expected, rotated_points, rtol=1e-5, atol=1e-8)), """A rotation by pi/2 should leave the z 
                                                                                coordinates invariant, while exhanging x and y
                                                                                coordinates and multplying y coordinates by -1."""

    def test_apply_rotation_preserves_dot_product(self):
        batch_size = 128
        n_points = 1006
        points1 = torch.randn(size=(batch_size, n_points, 3))
        points2 = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = roma.random_rotmat(batch_size)
        rotated1 = apply_rotation(points1, rotation_matrices)
        rotated2 = apply_rotation(points2, rotation_matrices)
        dot_prod1 = (points1*points2).sum(-1)
        dot_prod2 = (rotated1*rotated2).sum(-1)
        assert torch.all(torch.isclose(dot_prod1, dot_prod2, rtol=1e-4, atol=1e-6)), """ The apply_rotation function should 
                                                                                preserve the dot product."""

    def test_apply_rotation_gradients(self):
        batch_size = 128
        n_points = 1006
        points1 = torch.randn(size=(batch_size, n_points, 3))
        rotation_matrices = roma.random_rotmat(batch_size)
        rotation_matrices.requires_grad = True
        rotated = apply_rotation(points1, rotation_matrices)
        out = rotated.sum()
        out.backward()
        assert rotation_matrices.grad is not None, """The apply_rotation should carry the require_gradient property
                                                        of the rotation matrix."""
class TestApplyTranslation():
    def test_apply_translation_shape(self):
        points = torch.randn(size= (128, 1006, 3))
        translation_vectors = torch.randn(size=(128, 1006, 3))
        translated_points = apply_translation(points, translation_vectors)
        assert translated_points.shape == (128, 1006, 3)

    def test_apply_translation_identity(self):
        points = torch.randn(size= (128, 1006, 3))
        translation_vectors = torch.zeros(size=(128, 1006, 3))
        translated_points = apply_translation(points, translation_vectors)
        assert torch.all(torch.isclose(translated_points, points)), """Identity translation shoud leave the set of points
                                                                        invariant."""


    def test_apply_translation_gradient(self):
        points = torch.randn(size= (128, 1006, 3))
        translation_vectors = torch.zeros(size=(128, 1006, 3), requires_grad=True)
        translated_points = apply_translation(points, translation_vectors)
        loss = translated_points.mean()
        loss.backward()
        assert translation_vectors.grad is not None, """The gradient of the translation_vectors should carry the gradient"""

    def test_apply_translation_norm(self):
        points = torch.randn(size=(128, 1006, 3))
        translation_vectors = torch.zeros(size=(128, 1006, 3), requires_grad=True)
        translated_points = apply_translation(points, translation_vectors)
        norm_translated_points = (translated_points**2).sum(dim=-1)
        norm_points = (points**2).sum(dim=-1)
        translation_norm = (translation_vectors**2).sum(dim=-1)
        expected_norm = translation_norm + norm_points + 2*(translation_vectors*points).sum(dim=-1)
        assert torch.all(torch.isclose(expected_norm, norm_translated_points)), """The expected and translated norms do not
                                                                                    match"""
