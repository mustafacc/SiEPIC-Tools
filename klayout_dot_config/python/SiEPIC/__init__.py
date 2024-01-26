"""
SiEPIC-Tools package for KLayout
"""
import logging
import pya

logging.basicConfig(
    filename="siepic_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

__version__ = "0.6.0"

logging.info(f"KLayout SiEPIC-Tools version {__version__}")

if "__version__" in dir(pya):
    # pya.__version__ was introduced in KLayout version 0.28.6
    KLAYOUT_VERSION = int(pya.__version__.split(".")[1])
else:
    KLAYOUT_VERSION = int(pya.Application.instance().version().split(".")[1])
    KLAYOUT_VERSION_3 = int(pya.Application.instance().version().split(".")[2])

if KLAYOUT_VERSION < 28:
    # pya.Technologies was introduced in 0.27: https://www.klayout.de/doc-qt5/code/class_Library.html#method24
    # SiEPIC-Tools is being updated to use this functionality, hence will no longer be supported for KLayout 0.26
    error_msg = "\nSiEPIC-Tools is no longer compatible with older versions (0.27) of KLayout.\nPlease download an install the latest version from www.klayout.de"
    logging.error(error_msg)
    raise Exception(error_msg)

else:
    from . import _globals

    logging.info("SiEPIC Tools executing in mode: {_globals.Python_Env}")
    if _globals.Python_Env == "KLayout_GUI":
        from . import (
            extend,
            _globals,
            core,
            github,
            scripts,
            utils,
            setup,
            install,
            verification,
        )
    else:
        from . import extend, _globals, verification
