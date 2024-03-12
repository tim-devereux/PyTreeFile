import pandas as pd 

def treeinfo_attributes_tree(tree_file):
    """
    Extracts per-tree tree attributes from a tree info file and returns a DataFrame. 
    Can be run on a 'forest' or single treefile.

    Parameters:
    tree_file (str): The path to the treefile.

    Returns:
    pandas.DataFrame: A DataFrame containing tree attributes. If the input file is a forest, the DataFrame will contain a column 'tree_id' with the tree ID.
    """
    line_list = []
    with open(tree_file, 'r') as file:
        lines = file.readlines()
        line_count = 0        
        for line in lines:
            data = line.split(', ')
            for row in data:
                section_data = row.strip().split(', ')
                cell_data = section_data[0].strip().split(',')
                if len(cell_data) == 7:
                    line_list.append(cell_data)
            line_count += 1
    df = pd.DataFrame(line_list[1:], columns=line_list[0]).astype(float)
    if len(df) > 1:
        df.insert(0, 'tree_id', range(1, len(df) + 1))
        

    return df

def treeinfo_attributes_segment(tree_file):
    """
    Extracts per-segment attributes of a tree info file from a file and returns a DataFrame.
    Can be run on a 'forest' or single treefile.

    Parameters:
    tree_file (str): The path to the treefile.

    Returns:
    pandas.DataFrame: A DataFrame containing segment attributes. If the input file is a forest, the DataFrame will contain a column 'tree_id' with the tree ID.
    """
    line_list = []
    tree_ids = []
    tree_id = 0
    
    with open(tree_file, 'r') as file:
        lines = file.readlines()
        line_count = 0        
        for line in lines:
            data = line.split(', ')
            for row in data:
                section_data = row.strip().split(', ')
                cell_data = section_data[0].strip().split(',')
                if len(cell_data) == 7 and all(x.replace('.', '', 1).isdigit() for x in cell_data):
                    tree_id += 1
                if len(cell_data) == 14:
                    if tree_id != 0:
                        tree_ids.append(tree_id)
                    line_list.append(cell_data)
            line_count += 1
    df = pd.DataFrame(line_list[1:], columns=line_list[0]).astype(float)
    df.insert(0, 'tree_id', tree_ids)
    # remove row where parent_id is -1.0
    df = df[df['parent_id'] != -1.0]
    return df
