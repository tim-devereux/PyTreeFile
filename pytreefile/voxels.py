import pandas as pd


def read_voxels(filename):
    """
    Reads a voxel map from a file and returns a DataFrame.

    Parameters:
    filename (str): The path to the file.

    Returns:
    pandas.DataFrame: A DataFrame containing the voxel map.
    """
    with open(filename, "r") as file:
        lines = file.readlines()
        line_list = []
        for line in lines:
            data = line.split()
            line_list.append(data)
    df = pd.DataFrame(line_list[1:], columns=line_list[0]).astype(float)
    return df


def write_voxels(filename, df):
    """
    Writes a DataFrame to a file as a voxel map.

    Parameters:
    filename (str): The path to the file.
    df (pandas.DataFrame): The DataFrame to write.
    """
    with open(filename, "w") as file:
        file.write(df.to_string(index=False))
    return None


def points_mask_voxels(voxels_df, points_df):
    """
    Masks a voxel map with a point cloud and returns a DataFrame containing only the voxels containing points.

    Parameters:
    voxels_df (pandas.DataFrame): The voxel map.
    points_df (pandas.DataFrame): The point cloud.

    Returns:
    pandas.DataFrame: A DataFrame containing only the voxels containing points.
    """
    # Create an empty list to store voxels with points
    voxels_with_points = []

    # Iterate through each voxel
    for index, voxel in voxels_df.iterrows():
        # Extract voxel coordinates
        voxel_coords = voxel[["X", "Y", "Z"]].values

        # Extract voxel width
        voxel_width = voxel["VOX_WIDTH"]

        # Calculate voxel boundaries
        voxel_min = voxel_coords - voxel_width / 2
        voxel_max = voxel_coords + voxel_width / 2

        # Check if any point falls inside the voxel
        points_inside_voxel = points_df[
            (points_df["X"] >= voxel_min[0])
            & (points_df["X"] <= voxel_max[0])
            & (points_df["Y"] >= voxel_min[1])
            & (points_df["Y"] <= voxel_max[1])
            & (points_df["Z"] >= voxel_min[2])
            & (points_df["Z"] <= voxel_max[2])
        ]

        # If there are points inside the voxel, add the voxel to the list
        if not points_inside_voxel.empty:
            voxels_with_points.append(voxel)

    # Convert the list of voxels with points to a DataFrame
    voxels_with_points_df = pd.DataFrame(voxels_with_points)

    return voxels_with_points_df
