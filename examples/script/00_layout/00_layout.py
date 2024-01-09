# %% package imports
import os
import gsip
from SiEPIC.utils.layout import new_layout, floorplan
from SiEPIC.scripts import export_layout

if __name__ == '__main__':
    # create an empty layout with only a floorplan
    topcell, ly = new_layout(tech=gsip.name, topcell_name="TOP", overwrite=True)
    floorplan(topcell, 605e3, 410e3)

    # export the layout
    fpath = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.basename(fpath).removesuffix(".py")
    export_layout(topcell, fpath, fname, format="oas", screenshot=True)
