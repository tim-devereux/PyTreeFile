import os
import tempfile

import pandas as pd
import pytest

from pytreefile.points import read_points, write_points

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "example_data")


class TestReadWritePoints:
    def test_read_voxels_as_points(self):
        """Voxel files are whitespace-delimited, same format as points."""
        filepath = os.path.join(DATA_DIR, "dense_voxels.txt")
        df = read_points(filepath)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_roundtrip(self):
        df = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0], "Z": [7.0, 8.0, 9.0]}
        )
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            tmppath = f.name
        try:
            write_points(tmppath, df)
            df2 = read_points(tmppath)
            pd.testing.assert_frame_equal(
                df.reset_index(drop=True),
                df2.reset_index(drop=True),
                atol=1e-4,
            )
        finally:
            os.unlink(tmppath)
