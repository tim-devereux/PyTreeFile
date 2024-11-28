import open3d as o3d
import numpy as np
import pymeshlab
import os


def direction_vector(point1, point2):
    """
    Compute the direction vector between two points in 3D space.

    Parameters:
    - point1: tuple of floats (x1, y1, z1)
    - point2: tuple of floats (x2, y2, z2)

    Returns:
    - tuple of floats representing the direction vector
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return (x2 - x1, y2 - y1, z2 - z1)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2

    Parameters:
    vec1 (numpy.ndarray): The first vector.
    vec2 (numpy.ndarray): The second vector.

    Returns:
    numpy.ndarray: The rotation matrix.
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
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
    numpy.ndarray: An array containing the bin edges.
    numpy.ndarray: An array containing the number of points in each bin.
    """

    # Load point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_file)

    # Extract point coordinates
    points = np.asarray(pcd.points)

    # Get height range
    min_height = np.min(points[:, 1])
    max_height = np.max(points[:, 1])
    height_range = max_height - min_height

    # Define height bins
    num_bins = int(np.ceil(height_range / bin_size))
    bin_edges = np.linspace(min_height, min_height + num_bins * bin_size, num_bins + 1)

    # Count points in each bin
    counts, _ = np.histogram(points[:, 1], bins=bin_edges)

    return bin_edges, counts


def raycast_batch(start, end, scene, x, y, z):
    """
    Perform raycasting in a batch of points.

    Parameters:
    start (int): The start index of the batch.
    end (int): The end index of the batch.
    scene (open3d.t.geometry.RaycastingScene): The raycasting scene.
    x (numpy.ndarray): The x-coordinates of the points.
    y (numpy.ndarray): The y-coordinates of the points.
    z (numpy.ndarray): The z-coordinates of the points.

    Returns:
    numpy.ndarray: An array containing the points inside the mesh.
    """
    batch_points = np.array(
        [
            (x[i // (len(y) * len(z))], y[(i // len(z)) % len(y)], z[i % len(z)])
            for i in range(start, end)
        ]
    )
    query_points = o3d.core.Tensor(batch_points, dtype=o3d.core.Dtype.Float32)
    directions = np.tile(
        [0, 0, 1], (batch_points.shape[0], 1)
    )  # Assuming z-direction for rays
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
    numpy.ndarray: An array containing the bin edges.
    numpy.ndarray: An array containing the number of voxels in each bin.
    """

    # Here we use Pymeshlab to close the holes in the mesh, as open3d does not have a function to do this.
    # Create a new MeshSet
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_dir)
    ms.apply_filter("meshing_close_holes", maxholesize=30)
    ms.save_current_mesh(mesh_dir)

    # Now reload mesh into open3d to do the rest of the processing
    mesh = o3d.io.read_triangle_mesh(mesh_dir)

    # Clean the mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_vertex_normals()

    # Compute the AABB of the mesh
    aabb = mesh.get_axis_aligned_bounding_box()

    # Generate a grid of points for fine voxelization
    x = np.arange(aabb.min_bound[0], aabb.max_bound[0], voxel_size)
    y = np.arange(aabb.min_bound[1], aabb.max_bound[1], voxel_size)
    z = np.arange(aabb.min_bound[2], aabb.max_bound[2], voxel_size)

    # Prepare the mesh for raycasting
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    # Perform raycasting in parallel batches
    print("Computing volume profile for mesh: {}".format(mesh_dir))

    inside_points = np.vstack(
        [
            raycast_batch(i, min(i + batch_size, len(x) * len(y) * len(z)))
            for i in range(0, len(x) * len(y) * len(z), batch_size)
        ]
    )

    # Concatenate the results from all batches
    inside_points = inside_points if inside_points.size else np.array([])

    # Split the path into a root and extension
    root, _ = os.path.splitext(mesh_dir)

    # Save the inside points to a text file
    np.savetxt(root + "_inside_voxels.txt", inside_points)

    # Extract the Z coordinates
    z_coords = inside_points[:, 2]

    # Account for any negative Z-coordinates
    min_z = np.min(z_coords)
    if min_z < 0:
        z_coords -= min_z  # Shift to non-negative values

    z_bins = np.floor(z_coords / bin_size).astype(int)
    bin_counts = np.bincount(z_bins)

    # Adjust the bin_edges to reflect original z-coordinates for plotting
    bin_edges = np.arange(len(bin_counts)) * bin_size + (min_z - (min_z % bin_size))

    print("Volume profile computed for mesh: {}".format(mesh_dir))
    print("Bin counts: ", bin_counts)
    print("Bin edges: ", bin_edges)

    return np.array([bin_edges, (voxel_size * bin_counts)])
