import pandas as pd


def read_points(filename):
    """
    Reads a point cloud from a whitespace-delimited file and returns a DataFrame.

    Parameters:
    filename (str): The path to the file.

    Returns:
    pandas.DataFrame: A DataFrame containing the point cloud.
    """
    return pd.read_csv(filename, sep=r"\s+")


def write_points(filename, df):
    """
    Writes a DataFrame to a file as a point cloud.

    Parameters:
    filename (str): The path to the file.
    df (pandas.DataFrame): The DataFrame to write.
    """
    with open(filename, "w") as file:
        file.write(df.to_string(index=False))
