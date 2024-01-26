from pya import PCellDeclarationHelper, DPath, DPoint, Trans
from SiEPIC.utils import load_Waveguides_by_Tech


class Waveguide(PCellDeclarationHelper):
    def __init__(self):
        # Important: initialize the super class
        super(Waveguide, self).__init__()

        self.technology_name = "GSiP"

        # Load all strip waveguides
        self.waveguide_types = load_Waveguides_by_Tech(self.technology_name)

        # declare the parameters

        p = self.param(
            "waveguide_type",
            self.TypeList,
            "Waveguide Type",
            default=self.waveguide_types[0]["name"],
        )
        for wa in self.waveguide_types:
            p.add_choice(wa["name"], wa["name"])
        self.param(
            "path",
            self.TypeShape,
            "Path",
            default=DPath([DPoint(0, 0), DPoint(10, 0), DPoint(10, 10)], 0.5),
        )

        # self.param("length", self.TypeDouble, "Length", default = 0, readonly=True)
        """ todo - can add calculated values and parameters for user info
    self.param("radius", self.TypeDouble, "Bend Radius", default = 0, readonly=True)
    """

        self.cellName = "Waveguide"

    def display_text_impl(self):
        # Provide a descriptive text for the cell
        return "%s_%s" % (self.cellName, self.path)

    def coerce_parameters_impl(self):
        #    self.length = self.waveguide_length
        pass

    def can_create_from_shape_impl(self):
        return self.shape.is_path()

    def transformation_from_shape_impl(self):
        return Trans(Trans.R0, 0, 0)

    def parameters_from_shape_impl(self):
        self.path = self.shape.path

    def produce_impl(self):
        # https://github.com/KLayout/klayout/issues/879
        # tech = self.layout.library().technology

        # Make sure the technology name is associated with the layout
        #  PCells don't seem to know to whom they belong!
        if self.layout.technology_name == "":
            self.layout.technology_name = self.technology_name

        # Draw the waveguide geometry, new function in SiEPIC-Tools v0.3.90
        from SiEPIC.utils.layout import layout_waveguide4

        self.waveguide_length = layout_waveguide4(
            self.cell, self.path, self.waveguide_type, debug=False
        )
