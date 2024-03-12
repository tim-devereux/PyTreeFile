import open3d as o3d
import numpy as np
from geom import rotation_matrix_from_vectors, direction_vector

def recontruct_treefile(tree_df):
    """
    Reconstructs a tree from a treefile and returns a TriangleMesh.
    
    Parameters:
    tree_df (pandas.DataFrame): The treefile as a DataFrame.
    
    Returns:
    open3d.geometry.TriangleMesh: The reconstructed tree as a TriangleMesh.
    """
    # Create an empty mesh
    mesh = o3d.geometry.TriangleMesh()
    # Iterate through each row in the DataFrame
    for index, row in tree_df.iterrows():
        try:
            # Extract section_id and parent_id from the row data
            section_id = row['section_id']
            parent_id = row['parent_id']
            
            # Skip the row if it is the root section
            if parent_id == 0.0 or section_id == -1.0:
                continue

            parent_section = tree_df[tree_df['section_id'] == parent_id]
            start = np.array([parent_section.iloc[0, 0], parent_section.iloc[0, 1], parent_section.iloc[0, 2]])
            end = np.array([row[' x'], row['y'], row['z']])

            direction = direction_vector(start, end)
            length = np.abs(np.linalg.norm(end - start))
            radius = row['radius']

            # Create a mesh cylinder
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8, split=4)

            # Align the cylinder with the direction vector
            # The default direction of the cylinder is along the Z-axis (0, 0, 1)
            z_axis = np.array([0, 0, 1])
            R = rotation_matrix_from_vectors(z_axis, direction)
            cylinder.rotate(R, center=(0, 0, 0))

            # Translate the cylinder to the correct position
            cylinder.translate(start)

            # Combine with the main mesh
            mesh += cylinder
        except:
            print('error')
    return mesh

