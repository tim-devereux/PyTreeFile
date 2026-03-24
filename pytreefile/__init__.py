from .treefiles import treeinfo_attributes_tree, treeinfo_attributes_segment, attributes_tree
from .points import read_points, write_points


def __getattr__(name):
    """Lazy imports for modules with heavy dependencies (open3d, pymeshlab)."""
    if name == "direction_vector":
        from .geom import direction_vector
        return direction_vector
    if name == "rotation_matrix_from_vectors":
        from .geom import rotation_matrix_from_vectors
        return rotation_matrix_from_vectors
    if name == "count_points_in_height_bins":
        from .geom import count_points_in_height_bins
        return count_points_in_height_bins
    if name == "raycast_batch":
        from .geom import raycast_batch
        return raycast_batch
    if name == "compute_mesh_volume_profile":
        from .geom import compute_mesh_volume_profile
        return compute_mesh_volume_profile
    if name == "reconstruct_treefile":
        from .meshes import reconstruct_treefile
        return reconstruct_treefile
    raise AttributeError(f"module 'pytreefile' has no attribute {name!r}")
