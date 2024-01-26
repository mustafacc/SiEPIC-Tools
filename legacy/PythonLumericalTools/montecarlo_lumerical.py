# Circuit to simulate, which is found in a sub-folder with the same name

#circuit_name = "MZI"
#num_detectors = 1 # number of detectors to connect to the circuit

circuit_name = "MZI_bdc"
num_detectors = 2 # number of detectors to connect to the circuit

circuit_name = "RingResonator"
num_detectors = 3 # number of detectors to connect to the circuit

circuit_name = "tanner"
num_detectors = 1

# Find path to the example netlist files
import os, inspect
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
folder = os.path.join(path,circuit_name)
folder = "/Users/lukasc/Desktop/tanner"
if not os.path.exists(folder):
    folder = os.path.join(path,"..",circuit_name)
print(folder)

# make sure that the project folder is in the Python path
import sys
if not path in sys.path:
  sys.path.append(path)

# Setup Lumerical-Python integration, and load the SiEPIC-Tools Lumerical functions
import lumerical
import lumerical.load_lumapi
import lumerical.interconnect
lumapi = lumerical.load_lumapi.LUMAPI

# for debugging, to reload the lumerical module:
if 0:
    import sys
    if int(sys.version[0]) > 2:
      from importlib import reload
    reload(lumerical.interconnect)
    reload(lumerical.load_lumapi)


# Start Lumerical INTERCONNECT
lumerical.interconnect.run_INTC()
INTC = lumerical.interconnect.INTC
lumapi.evalScript(INTC, "?'Test';")


# Perform Lumerical INTERCONNECT simulation

# Regular simulation:
if 1:
  lumerical.interconnect.circuit_simulation(circuit_name=circuit_name, folder=folder, num_detectors=num_detectors, matlab_data_files=[], simulate=True, verbose=False)

# Monte Carlo simulation:
if 0:
  lumerical.interconnect.circuit_simulation_monte_carlo(circuit_name=circuit_name, folder=folder, num_detectors=num_detectors, matlab_data_files=[], simulate=True, verbose=False)


