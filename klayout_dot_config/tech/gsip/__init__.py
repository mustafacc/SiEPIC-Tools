# Load the KLayout technology, when running in Script mode
from SiEPIC.utils import get_technology_by_name
import pya, os
name = 'GSiP'

print(f'SiEPIC PDK Python module: KLayout technology: {name}')

try:
    tech = pya.Technology().create_technology(name)
    tech.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), f'{name}.lyt'))
except Exception as e:
    tech = get_technology_by_name(name)
    print(f"Technology {name} already loaded.")
# then import all the technology modules
from . import pymacros

