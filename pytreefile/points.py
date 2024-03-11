import pandas as pd

def read_points(filename):
    """
    Reads a point cloud from a file and returns a DataFrame.

    Parameters:
    filename (str): The path to the file.

    Returns:
    pandas.DataFrame: A DataFrame containing the point cloud.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        line_list = []
        for line in lines:
            data = line.split()
            line_list.append(data)
    df = pd.DataFrame(line_list[1:], columns=line_list[0]).astype(float)
    return df


def write_points(filename, df):
    """
    Writes a DataFrame to a file as a point cloud.

    Parameters:
    filename (str): The path to the file.
    df (pandas.DataFrame): The DataFrame to write.
    """
    with open(filename, 'w') as file:
        file.write(df.to_string(index=False))
    return None