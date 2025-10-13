.. _tutorialCBTE:

Coupled BTE Tutorial
=========================

Overview
--------

In order to predict thermoelectric transport properties including the phonon drag effect, we have to construct the coupled electron-phonon Boltzmann Transport Equation (epBTE). We do this using a full scattering matrix approach, so that the coupled scattering matrix consists of the standard electron and phonon scattering matrices, along with off diagonal drag terms, which couple the previously independent electron and phonon subspaces. Then, we solve the CBTE using the relaxons solution. 

Before proceeding to the CBTE tutorial, we strongly recommend you first read the :ref:`theoryCBTE` section of the theory documentation, and should also review the :ref:`relaxons` to see how the standard electron and phonon relaxons calculations are used. 

.. note::
  We note that this is an extreme computational effort -- both in memory and compute requirements, as many scattering rates are required. 
    
.. important:: 
  At this time, we consider it a beta feature -- if you encounter difficulty, please post this on the `discussions page <https://github.com/phoebe-team/phoebe/discussions>`_ of the Phoebe GitHub repository. 

Step 0: Preparing the input files 
----------------------------------

As in the case of the :ref:`phononElectronTransport`, you will need electron and phonon input files from both the :ref:`elWanTransport` and :ref:`phononTransport` calculations (making sure to use the same crystal structure in both el and ph file generation). 
These files can be taken directly from the output of the el and ph transport tutorials, which generate files for silicon, as in the example `here <https://github.com/phoebe-team/phoebe/tree/develop/example/Silicon-coupled>`_:

+---------------------------+-----------------------------+  
|**For electrons:**         |        **For phonons:**     |
+===========================+=============================+
| From Quantum ESPRESSO:    | From phono3py:              | 
|  - ``*.phoebe.elph.hdf5`` |   - ``fc2.hdf5``            |
|  - ``*.fc``               |   - ``fc3.hdf5``            |
|  - ``*_tb.dat``           |   - ``phono3py_disp.yaml``  |
|                           | Or from shengBTE:           |
|                           |   - ``*.fc``                |
|                           |   - ``FORCE_CONSTANTS_3RD`` |
+---------------------------+-----------------------------+  

With these in hand, we can proceed to perform a coupled BTE calculation. For this tutorial, we will perform a calculation of doped silicon, so ``* = si`` or ``silicon``.

Step 1: Running the Coupled BTE calculation 
-------------------------------------------

This calculation will construct a scattering matrix using electron-phonon, phonon-phonon, phonon-isotope, phonon-electron, and drag term scattering rates. 
We can set up a coupled BTE calculation using the following example input file, as currently exists in ``phoebe/examples/Silicon-coupled/coupledTransport.in``:: 
    
  appName = "coupledTransport"
  
  sumRuleFC2 = "crystal"
  phFC2FileName = "silicon.fc"
  electronH0Name = "si_tb.dat"
  elphFileName = "silicon.phoebe.elph.hdf5"
  phFC3FileName = "FORCE_CONSTANTS_3RD"

  kMesh = [25, 25, 25]
  qMesh = [5, 5, 5]
  temperatures = [200.]
  dopings = [1.e21]

  smearingMethod = "gaussian"
  elSmearingWidth = 0.005 eV
  phSmearingWidth = 0.002 eV  
  windowType = "population"
  windowPopulationLimit = 1e-3
  numOccupiedStates = 4

  useSymmetries = false
  enforceDetailedBalance = true 
  symmetrizeMatrix = true
  scatteringMatrixInMemory = true
  solverBTE = ["relaxons"]
  
| Most parameters used here are applicable to all relaxon calculations and are described in the relaxons tutorial. 
| **Here we address a few points specific to the coupled BTE calculation:**

  - The population window limit and k/q-grids here are *very* coarse. One should increase them and the grids used in the calculation until it is converged. 
  - We have chosen a ``qMesh`` which is commensurate with our ``kMesh``. This is required in the coupled BTE calculation so that the same electron and phonon states are used in the electron-phonon, phonon-electron, and drag contributions to the scattering matrix. 
  - As with all relaxons calculations, we here choose Gaussian smearing. However now, we have ``elSmearingWidth`` and ``phSmearingWidth`` listed separately. This is required because electron and phonon energy scales are dramatically different, resulting in different requirements for mesh samplings and as a result differences in smearing values. ``phSmearingWidth`` will apply to phonon-phonon and phonon-isotope scattering, and the ``elSmearingWidth`` applies to electron-phonon, phonon-electron, and drag terms. 
  - Because the coupled BTE calculation is very sensitive to interpolation error with respect to the quality of the electron-phonon matrix elements, it's very likely that we will need to apply ``enforceDetailedBalance = true`` to enforce that the diagonal of the scattering matrix can be reconstructed from the off-diagonal scattering rates (detailed balance).
  - Note, as with earlier electron and phonon only relaxons solutions, here the use of symmetries is still a research problem, so we have to have ``useSymmetries = false``.

.. note::
  A very important part of this calculation is that it's extremely easy to come up with negative eigenvalues. 
  Please check carefully the appendix section of the relevant paper by Coulter et al. for discussion about the positive semi-definite nature of the coupled scattering matrix.

This should be run just as in the other tutorial::

  export OMP_NUM_THREADS=4
  mpirun -np 4 /path/to/phoebe/build/phoebe -in coupledTransport.in > coupledTransport.out
  
We note that this one can take a bit of time, so we recommend using a cluster or workstation for the demonstration. 
The parallelism of this calculation follows the same principles as that of the standard relaxons solution as discussed in :ref:`relaxonsParallelism`.

Output and Post-Processing 
--------------------------

As in the standard relaxons solution to the BTE, we have outputs containing bulk transport coefficients, including:
  
  * ``relaxons_coupled_viscosity.json``
  * ``relaxons_coupled_transport_coefficients.json``
  * ``relaxons_coupled_relaxation_times.json``
  
These files contain the viscosity, electrical and thermal conductivity, and relaxation times calculated from the coupled relaxons solution in ``json`` format. 

As before, we will also have the files ``relaxons_el_eigenvectors.hdf5`` or ``relaxons_ph_eigenvectors.hdf5``, which contain the el and ph contributions to each coupled relaxon eigenvector of the scattering matrix. 
These again can be plotted on the Wigner-Seitz cell using the script provided in ``phoebe/plotScripts/relaxons_eigenvectors.py``.

Finally, the contents of ``relaxons_coupled_real_space_coefficients.json`` can be used to parameterize the Viscous Thermoelectric Equations as in Coulter, Rajkov, and Simoncelli (2025). `arXiv:2503.07560 <https://arxiv.org/abs/2503.07560>`_ 
