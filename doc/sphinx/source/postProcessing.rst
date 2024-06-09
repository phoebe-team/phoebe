.. _postprocessing:

Results Postprocessing
======================

JSON Output Files
-----------------

The majority of output files produced by Phoebe will be in the JSON format. JSON files can be thoughtof as dictionaries. If you process results with Python, for example::

  import json

  # open and parse the file
  f = open("epa_relaxation_times.json","r")
  resultsDict = json.load(f)

  # print the keys of the dictionary to see what the file contains
  for key, value in resultsDict.items():
    print(key)

This opens a JSON file and prints the keys to the dictionary. The keys indicate which data sets are available in a particular JSON file. We store transport/lifetime data, particle type, band energies, dos, as well as output units in these files.

Plotting Scripts
-----------------

Additionally, in the subfolder ``phoebe/scripts``, we prepared a few simple python scripts that can be used to plot the results in JSON files.

The scripts are self-documented, e.g. you can run::

  python bands.py -h

to see a description of the script.
In general, these scripts have at least one input argument that must be passed on the command line, including the path to the suitable JSON file.
The scripts generate an output ``pdf`` file at the same location as the input JSON file.

Feel free to use these scripts as a starting point for your own customized plots. If you write an additional script which you think is valuable to other users, consider letting us know and we may add it to the repository.


bands.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots the band structure. Example use::

  python bands.py electron_bands.json


transport_coefficients.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots the transport coefficients (thermal conductivity, electrical conductivity, etc...) as a function of temperature.
Works for transport coefficients computed with different BTE solvers (EPA, Wannier interpolation, RTA, iterative, ...).
The arguments here are the direction of transport and also the the x-axis type, which can be either doping (n) or temperature (T).
Example use::

  python transport_coefficients.py rta_phonon_thermal_cond.json n xx

or::

  python transport_coefficients.py epa_onsager_coefficients.json T xx


dos.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots the density of states. Example use::

  python dos.py electron_dos.json


tau_path.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Overlays the band structure with the particle linewidths, using the output of the lifetimes apps. Example use::

  python tau_path.py path_el_relaxation_times.json path_el_bandstructure.json 0

In this case, we follow the json file with 0 because this script takes a "calculation index". The calculation index is 0 unless you used multiple temperatures or dopings. If this is the case, you want to supply the calculation index corresponding to the doping/temperature you want to plot.

tau.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots lifetimes vs energy. Example usage::

  python tau.py rta_relaxation_times.json 0

See note above under ``tau_path.py`` for more info about calcIndex (the 0 at the end).

epa_tau.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots lifetimes vs energy from an EPA calculation. Example usage::

  python epa_tau.py epa_relaxation_times.json 0

See note above under ``tau_path.py`` for more info about calcIndex (the 0 at the end).

tau_bandResolved.py
^^^^^^^^^^^^^^^^^^^^^^^^^

Plots lifetimes/linewidths vs energy. It's similar to the script tau.py, but it has a color-code for lifetimes and linewidths from each different band and it only works if no window has been used in the transport code. Note that the n-th band index is defined as the n-th lowest-energy band. Example use::

  python tau_bandResolved.py rta_relaxation_times.json 0

See note above under ``tau_path.py`` for more info about calcIndex (the 0 at the end).
