import logging

import numpy as np

from .geom import rotation_matrix_from_vectors, direction_vector

logger = logging.getLogger(__name__)


def reconstruct_treefile(tree_df):
    """
    Reconstructs a tree from a treefile and returns a TriangleMesh.

    Parameters:
    tree_df (pandas.DataFrame): The treefile as a DataFrame.

    Returns:
    open3d.geometry.TriangleMesh: The reconstructed tree as a TriangleMesh.
    """
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()

    for index, row in tree_df.iterrows():
        section_id = row["section_id"]
        parent_id = row["parent_id"]

        if parent_id == 0.0 or section_id == -1.0:
            continue

        parent_section = tree_df[tree_df["section_id"] == parent_id]
        if parent_section.empty:
            logger.warning("No parent found for section_id=%s, parent_id=%s", section_id, parent_id)
            continue

        start = np.array(
            [
                parent_section.iloc[0, 0],
                parent_section.iloc[0, 1],
                parent_section.iloc[0, 2],
            ]
        )
        end = np.array([row[" x"], row["y"], row["z"]])

        direction = direction_vector(start, end)
        length = np.linalg.norm(end - start)
        if length == 0:
            continue

        radius = row["radius"]

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=8, split=4
        )

        z_axis = np.array([0, 0, 1])
        R = rotation_matrix_from_vectors(z_axis, direction)
        cylinder.rotate(R, center=(0, 0, 0))
        cylinder.translate(start)

        mesh += cylinder

    return mesh
