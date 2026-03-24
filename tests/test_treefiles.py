import os
import pytest
import pandas as pd

from pytreefile.treefiles import (
    treeinfo_attributes_tree,
    treeinfo_attributes_segment,
    attributes_tree,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "example_data")


class TestTreeinfoAttributesTree:
    def test_single_tree(self):
        filepath = os.path.join(DATA_DIR, "tree_info.txt")
        df = treeinfo_attributes_tree(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        # Single tree should have 7 columns (no tree_id for single row)
        assert all(
            col in df.columns
            for col in [
                "height",
                "crown_radius",
                "dimension",
                "monocotal",
                "DBH",
                "bend",
                "branch_slope",
            ]
        )

    def test_forest(self):
        filepath = os.path.join(DATA_DIR, "forest_trees_info.txt")
        df = treeinfo_attributes_tree(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1
        assert "tree_id" in df.columns
        assert df["tree_id"].iloc[0] == 1

    def test_values_are_float(self):
        filepath = os.path.join(DATA_DIR, "tree_info.txt")
        df = treeinfo_attributes_tree(filepath)
        for col in df.columns:
            assert df[col].dtype == float


class TestTreeinfoAttributesSegment:
    def test_single_tree(self):
        filepath = os.path.join(DATA_DIR, "tree_info.txt")
        df = treeinfo_attributes_segment(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "tree_id" in df.columns
        # Should not contain root segments
        assert (df["parent_id"] != -1.0).all()

    def test_forest(self):
        filepath = os.path.join(DATA_DIR, "forest_trees_info.txt")
        df = treeinfo_attributes_segment(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "tree_id" in df.columns
        # Multiple trees should have different tree_ids
        assert df["tree_id"].nunique() > 1


class TestAttributesTree:
    def test_raycloud_trees(self):
        filepath = os.path.join(DATA_DIR, "raycloud_trees.csv")
        df = attributes_tree(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        # Should have no duplicate rows
        assert len(df) == len(df.drop_duplicates())

    def test_values_are_numeric(self):
        filepath = os.path.join(DATA_DIR, "raycloud_trees.csv")
        df = attributes_tree(filepath)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col])
