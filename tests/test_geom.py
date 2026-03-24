import numpy as np
import pytest

from pytreefile.geom import direction_vector, rotation_matrix_from_vectors


class TestDirectionVector:
    def test_basic(self):
        result = direction_vector((0, 0, 0), (1, 2, 3))
        assert result == (1, 2, 3)

    def test_negative(self):
        result = direction_vector((1, 2, 3), (0, 0, 0))
        assert result == (-1, -2, -3)

    def test_same_point(self):
        result = direction_vector((5, 5, 5), (5, 5, 5))
        assert result == (0, 0, 0)


class TestRotationMatrixFromVectors:
    def test_identity(self):
        """Same direction should give identity matrix."""
        v = np.array([1.0, 0, 0])
        R = rotation_matrix_from_vectors(v, v)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_opposite(self):
        """Opposite direction should give 180-degree rotation."""
        v1 = np.array([1.0, 0, 0])
        v2 = np.array([-1.0, 0, 0])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        np.testing.assert_array_almost_equal(result, v2)

    def test_perpendicular(self):
        """X to Y should work correctly."""
        v1 = np.array([1.0, 0, 0])
        v2 = np.array([0, 1.0, 0])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ v1
        np.testing.assert_array_almost_equal(result, v2)

    def test_arbitrary(self):
        """Arbitrary vectors should produce valid rotation."""
        v1 = np.array([1.0, 2, 3])
        v2 = np.array([4.0, 5, 6])
        R = rotation_matrix_from_vectors(v1, v2)
        result = R @ (v1 / np.linalg.norm(v1))
        expected = v2 / np.linalg.norm(v2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_is_rotation_matrix(self):
        """Result should be a proper rotation matrix (det=1, R^T R = I)."""
        v1 = np.array([1.0, 2, 3])
        v2 = np.array([4.0, 5, 6])
        R = rotation_matrix_from_vectors(v1, v2)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0)
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3))

    def test_z_to_z(self):
        """Z-axis to Z-axis (common in cylinder alignment)."""
        z = np.array([0, 0, 1.0])
        R = rotation_matrix_from_vectors(z, z)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_z_to_negative_z(self):
        """Z-axis to -Z-axis."""
        z = np.array([0, 0, 1.0])
        neg_z = np.array([0, 0, -1.0])
        R = rotation_matrix_from_vectors(z, neg_z)
        result = R @ z
        np.testing.assert_array_almost_equal(result, neg_z)
