# %%
import pytest
import gsip
from SiEPIC.utils.layout import new_layout
from SiEPIC.scripts import instantiate_all_library_cells


@pytest.mark.parametrize("tech", ["GSiP"])
def test_all_library_cells(tech):
    # Create a new layout
    topcell, ly = new_layout(tech, "UnitTesting", overwrite=True)

    # Instantiate all cells
    instantiate_all_library_cells(topcell)

    # Check if there are any errors or empty cells
    for cell_id in topcell.called_cells():
        c = ly.cell(cell_id)
        error_shapes = c.shapes(ly.error_layer())
        for error in error_shapes.each():
            assert False, f"Error in cell: {c.name}, {error.text}"
        assert not c.is_empty(), f"Empty cell: {c.name}"
        assert c.bbox().area() != 0


if __name__ == "__main__":
    test_all_library_cells(tech=gsip.name)
# %%
