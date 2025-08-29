.. _tutorialCBTE:

Coupled BTE Tutorial
=========================

Step 0: Preparing the input files 
----------------------------------

Before proceeding to the coupled BTE tutorial, we recommend you first read the :ref:`theoryCBTE` section of the theory documentation, and should also review the :ref:`relaxons` to see how the standard electron and phonon relaxons calculations are used. 
You will also need electron and phonon input files from both the :ref:`elWanTransport` and :ref:`phononTransport` calculations (making sure to use the same crystal structure in both el and ph file generation), including:  

+---------------------------+-----------------------------+  
|**For electrons:**         | **For phonons:**            |
+===========================+=============================+
| From Quantum ESPRESSO:    | From phono3py:              | 
|  - ``*.phoebe.elph.hdf5`` |   - ``fc2.hdf5``            |
|  - ``*.fc``               |   - ``fc3.hdf5``            |
|  - ``*_tb.dat``           | Or from shengBTE:           |
|                           |   - ``*.fc``                |
|                           |   - ``FORCE_CONSTANTS_3RD`` |
+---------------------------+-----------------------------+  

With these in hand, we can proceed to perform a coupled BTE calculation. 

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
  
Where these parameters were described in the relaxons tutorial. 
However, there are a few specific points which need to be discussed, which are specific to the coupled BTE calculation. 

  - The population window limit and k/q-grids here are _very_ coarse. One should increase them and the grids used in the calculation until it is converged. 
  - We have chosen a ``qMesh`` which is commensurate with our ``kMesh``. This is required in the coupled BTE calculation so that the same electron and phonon states are used in the electron-phonon, phonon-electron, and drag contributions to the scattering matrix. 
  - As with all relaxons calculations, we here choose Gaussian smearing. However now, we have ``elSmearingWidth`` and ``phSmearingWidth`` listed separately. This is required because electron and phonon energy scales are dramatically different, resulting in different requirements for mesh samplings and as a result differences in smearing values. 
  the ``phSmearingWidth`` will apply to phonon-phonon and phonon-isotope scattering, and the ``elSmearingWidth`` applies to electron-phonon, phonon-electron, and drag terms. 
  - Because the coupled BTE calculation is very sensitive to interpolation error with respect to the quality of the electron-phonon matrix elements, it's very likely that we will need to apply ``enforceDetailedBalance = true`` to enforce detailed balance for the matrix.
  - Note, as with earlier electron and phonon only relaxons solutions, here the use of symmetries is still a research problem, so we have to have ``useSymmetries = false``.

  