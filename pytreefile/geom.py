import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def direction_vector(point1, point2):
    """
    Compute the direction vector between two points in 3D space.

    Parameters:
    point1: array-like of floats (x1, y1, z1)
    point2: array-like of floats (x2, y2, z2)

    Returns:
    tuple of floats representing the direction vector
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return (x2 - x1, y2 - y1, z2 - z1)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2.

    Parameters:
    vec1 (numpy.ndarray): The source vector.
    vec2 (numpy.ndarray): The target vector.

    Returns:
    numpy.ndarray: The 3x3 rotation matrix.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-10:
        # Vectors are parallel — return identity or 180-degree rotation
        if c > 0:
            return np.eye(3)
        # 180-degree rotation: pick an arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(a, perp)
        u = u / np.linalg.norm(u)
        return 2 * np.outer(u, u) - np.eye(3)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def count_points_in_height_bins(point_cloud_file, bin_size):
    """
    Count the number of points in each height bin of a point cloud.

    Parameters:
    point_cloud_file (str): The path to the point cloud file.
    bin_size (float): The size of the height bins.

    Returns:
    tuple: (bin_edges, counts) arrays
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(point_cloud_file)
    points = np.asarray(pcd.points)

    min_height = np.min(points[:, 1])
    max_height = np.max(points[:, 1])
    height_range = max_height - min_height

    num_bins = int(np.ceil(height_range / bin_size))
    bin_edges = np.linspace(min_height, min_height + num_bins * bin_size, num_bins + 1)

    counts, _ = np.histogram(points[:, 1], bins=bin_edges)

    return bin_edges, counts


def raycast_batch(start, end, scene, x, y, z):
    """
    Perform raycasting on a batch of grid points.

    Parameters:
    start (int): The start index of the batch.
    end (int): The end index of the batch.
    scene (open3d.t.geometry.RaycastingScene): The raycasting scene.
    x (numpy.ndarray): The x-coordinates of the grid.
    y (numpy.ndarray): The y-coordinates of the grid.
    z (numpy.ndarray): The z-coordinates of the grid.

    Returns:
    numpy.ndarray: Points that are inside the mesh.
    """
    import open3d as o3d
    batch_points = np.array(
        [
            (x[i // (len(y) * len(z))], y[(i // len(z)) % len(y)], z[i % len(z)])
            for i in range(start, end)
        ]
    )
    query_points = o3d.core.Tensor(batch_points, dtype=o3d.core.Dtype.Float32)
    directions = np.tile([0, 0, 1], (batch_points.shape[0], 1))
    rays = np.concatenate(
        [query_points.numpy(), directions.astype(np.float32)], axis=-1
    )
    intersection_counts = scene.count_intersections(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ).numpy()
    return batch_points[intersection_counts % 2 == 1]


def compute_mesh_volume_profile(
    mesh_dir, voxel_size=0.05, bin_size=0.5, batch_size=10000
):
    """
    Compute the volume profile of a mesh using raycasting.

    Parameters:
    mesh_dir (str): The path to the mesh file.
    voxel_size (float): The size of the voxels.
    bin_size (float): The size of the height bins.
    batch_size (int): The size of the raycasting batches.

    Returns:
    numpy.ndarray: A 2-row array of [bin_edges, volumes].
    """
    import open3d as o3d
    import pymeshlab
    # Use Pymeshlab to close holes in the mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_dir)
    ms.apply_filter("meshing_close_holes", maxholesize=30)
    ms.save_current_mesh(mesh_dir)

    # Reload mesh into open3d
    mesh = o3d.io.read_triangle_mesh(mesh_dir)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    aabb = mesh.get_axis_aligned_bounding_box()

    x = np.arange(aabb.min_bound[0], aabb.max_bound[0], voxel_size)
    y = np.arange(aabb.min_bound[1], aabb.max_bound[1], voxel_size)
    z = np.arange(aabb.min_bound[2], aabb.max_bound[2], voxel_size)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    total_points = len(x) * len(y) * len(z)
    logger.info("Computing volume profile for mesh: %s (%d grid points)", mesh_dir, total_points)

    batches = [
        raycast_batch(i, min(i + batch_size, total_points), scene, x, y, z)
        for i in range(0, total_points, batch_size)
    ]
    non_empty = [b for b in batches if len(b) > 0]

    if not non_empty:
        logger.warning("No interior points found for mesh: %s", mesh_dir)
        return np.array([[], []])

    inside_points = np.vstack(non_empty)

    root, _ = os.path.splitext(mesh_dir)
    np.savetxt(root + "_inside_voxels.txt", inside_points)

    z_coords = inside_points[:, 2]

    min_z = np.min(z_coords)
    if min_z < 0:
        z_coords = z_coords - min_z

    z_bins = np.floor(z_coords / bin_size).astype(int)
    bin_counts = np.bincount(z_bins)

    bin_edges = np.arange(len(bin_counts)) * bin_size + (min_z - (min_z % bin_size))

    logger.info("Volume profile computed for mesh: %s", mesh_dir)

    return np.array([bin_edges, voxel_size * bin_counts])
