# %% package imports
import os
import gsip
from SiEPIC.utils.layout import new_layout, floorplan
from SiEPIC.scripts import export_layout, connect_pins_with_waveguide
from SiEPIC.extend import to_itype
from pya import Trans, CellInstArray, Text

# %% create a new layout with back-to-back GC
user = "Mustafa"
gc_cell = "Grating_Coupler_13deg_TE_1550_Oxide"
waveguide = "Strip TE 1550 nm"

topcell, ly = new_layout(tech=gsip.name, topcell_name="TOP", overwrite=True)
floorplan(topcell, to_itype(605), to_itype(410))

cell_gc = ly.create_cell(gc_cell, gsip.name)
x = to_itype(100)
y = to_itype(100)

t = Trans(Trans.R0, x, y)
instGC1 = topcell.insert(CellInstArray(cell_gc.cell_index(), t))

y += to_itype(127)
t = Trans(Trans.R0, x, y)
instGC2 = topcell.insert(CellInstArray(cell_gc.cell_index(), t))

# automated test label
text = Text(f"opt_in_TE_1550_device_{user}_MZI1", t)
topcell.shapes(ly.layer(ly.TECHNOLOGY["Text"])).insert(text).text_size = 2 / ly.dbu

# connect GCs
connect_pins_with_waveguide(
    instGC1, "opt_wg", instGC2, "opt_wg", waveguide_type=waveguide
)

# %% export the layout
fpath = os.path.dirname(os.path.abspath(__file__))
fname = os.path.basename(fpath).removesuffix(".py")
export_layout(topcell, fpath, fname, format="oas", screenshot=True)

# %%
