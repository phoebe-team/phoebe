Input File Description
======================

Here we provide a description of the input variables related to each of the applications in the code.
Phoebe tries to check the variables provided in input, and stops the code if it detects missing or invalid input parameters.

Note that the executable, ``phoebe`` is the same for all applications. The applications are selected through the :ref:`appName` parameter.

.. note::
   The input file text is case sensitive!

-------------------

QE to Phoebe
------------

**Functionality:** Convert the electron-phonon coupling from Quantum-ESPRESSO format to Phoebe format.
This postprocesses the data for  Wannier interpolation or EPA calculations in Phoebe.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "elPhQeToPhoebe"

* :ref:`phFC2FileName`

* :ref:`elPhInterpolation`

* :ref:`electronH0Name`

* :ref:`wannier90Prefix`

* :ref:`quantumEspressoPrefix`

* :ref:`epaMinEnergy`

* :ref:`epaMaxEnergy`

* :ref:`epaNumBins`

* :ref:`epaSmearingEnergy`

* :ref:`distributedElPhCoupling`

* :ref:`hdf5ElPhFileFormat`

.. raw:: html

  <h3>Sample input file (Wannier interpolation)</h3>

::

  appName = "elPhQeToPhoebe"
  elPhInterpolation = "wannier"

  phFC2FileName = "./silicon.fc"
  electronH0Name = "./si_tb.dat",
  wannier90Prefix = "si"
  quantumEspressoPrefix = "silicon"

.. raw:: html

  <h3>Sample input file (EPA)</h3>

::

  appName = "elPhQeToPhoebe"
  elPhInterpolation = "epa"

  phFC2FileName = "./silicon.fc"
  electronH0Name = "./out/silicon.xml",
  quantumEspressoPrefix = "silicon"

  electronFourierCutoff = 4.
  epaMinEnergy = -4. eV
  epaMaxEnergy = 10. eV
  epaNumBins = 10
  epaSmearingEnergy = 0.05 eV


-----------------------------------

Phonon BTE Solver
----------------------

**Functionality:** Build and solve the phonon Boltzmann Transport Equation (BTE) to compute phonon transport properties such as thermal conductivity, relaxation times and viscosity.

.. raw:: html

  <h3>Input variables</h3>

:ref:`appName` = "phononTransport"

* :ref:`phFC2FileName`

* :ref:`phFC3FileName`

* :ref:`phonopyDispFileName`

* :ref:`phonopyBORNFileName`

* :ref:`sumRuleFC2`

* :ref:`qMesh`

* :ref:`temperatures`

* :ref:`minTemperature`

* :ref:`maxTemperature`

* :ref:`deltaTemperature`

* :ref:`smearingMethod`

* :ref:`smearingWidth`

* :ref:`solverBTE`

* :ref:`scatteringMatrixInMemory`

* :ref:`symmetrizeMatrix`

* :ref:`windowType`

* :ref:`windowEnergyLimit`

* :ref:`windowPopulationLimit`

* :ref:`maxIterationsBTE`

* :ref:`convergenceThresholdBTE`

* :ref:`dimensionality`

* :ref:`constantRelaxationTime`

* :ref:`withIsotopeScattering`

* :ref:`masses`

* :ref:`isotopeCouplings`

* :ref:`boundaryLength`

* :ref:`useSymmetries`

* :ref:`numRelaxonsEigenvalues`

* :ref:`checkNegativeRelaxons`

.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "phononTransport"

  phFC2FileName = "./ForceConstants2nd"
  phFC3FileName = "./ForceConstants3rd"
  sumRuleFC2 = "crystal"
  qMesh = [10,10,10]
  temperatures = [300.]
  smearingMethod = "adaptiveGaussian"
  solverBTE = ["variational","relaxons"]
  scatteringMatrixInMemory = true
  boundaryLength = 10. mum

  #if using RTA or iterative solvers only, uncomment this
  #useSymmetries = true

-------------------------------------

Electron BTE Solver
-----------------------------------

**Functionality:** Build and solve the electronic Boltzmann Transport Equation (BTE) using Wannier interpolation. Output quantites are electrical conductivity, electronic thermal conductivity, Seebeck coefficient, electron viscosity, and electronic lifetimes (on the kMesh used in this calculation).

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronWannierTransport"

* :ref:`phFC2FileName`

* :ref:`sumRuleFC2`

* :ref:`electronH0Name`

* :ref:`wsVecFileName`

* :ref:`elphFileName`

* :ref:`kMesh`

* :ref:`temperatures`

* :ref:`minTemperature`

* :ref:`maxTemperature`

* :ref:`deltaTemperature`

* :ref:`dopings`

* :ref:`chemicalPotentials`

* :ref:`minChemicalPotential`

* :ref:`maxChemicalPotential`

* :ref:`deltaChemicalPotential`

* :ref:`smearingMethod`

* :ref:`smearingWidth`

* :ref:`dimensionality`

* :ref:`constantRelaxationTime`

* :ref:`convergenceThresholdBTE`

* :ref:`maxIterationsBTE`

* :ref:`windowType`

* :ref:`windowEnergyLimit`

* :ref:`windowPopulationLimit`

* :ref:`solverBTE`

* :ref:`scatteringMatrixInMemory`

* :ref:`symmetrizeMatrix`

* :ref:`fermiLevel`

* :ref:`numOccupiedStates`

* :ref:`useSymmetries`

* :ref:`numRelaxonsEigenvalues`

* :ref:`checkNegativeRelaxons`


.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "electronWannierTransport"

  phFC2FileName = "./silicon.fc"
  electronH0Name = "./si_tb.dat",
  elphFileName = "silicon.phoebe.elph.dat"
  sumRuleFC2 = "crystal"
  kMesh = [15,15,15]
  temperatures = [300.]
  dopings = [1.e21]
  smearingMethod = "gaussian"
  smearingWidth = 0.5 eV
  windowType = "population"


-------------------------------------

EPA Transport
-----------------------------------

**Functionality:** Build and solve the electronic Boltzmann Transport Equation (BTE) using Wannier interpolation. Output quantites are electrical conductivity, electronic thermal conductivity, Seebeck coefficient, electron viscosity, and electronic lifetimes as a function of bin energy.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "transportEPA"

* :ref:`electronH0Name`

* :ref:`epaFileName`

* :ref:`electronFourierCutoff`

* :ref:`epaEnergyStep`

* :ref:`epaEnergyRange`

* :ref:`kMesh`

* :ref:`temperatures`

* :ref:`dopings`


.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "transportEpa"

  electronH0Name = "./out/silicon.xml",
  epaFileName = "./silicon.phoebe.epa.dat"

  electronFourierCutoff = 4.
  epaEnergyStep = 0.01 eV
  epaEnergyRange = 3.0 eV

  kMesh = [10,10,10]
  temperatures = [300.]
  dopings = [1.0e21]

-----------------------------------

Phonon Lifetimes on a Path
--------------------------

**Functionality:** Compute phonon lifetimes/linewidths on a specified path through the Brillouin zone.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "phononLifetimes"

* :ref:`phFC2FileName`

* :ref:`sumRuleFC2`

* :ref:`phFC3FileName`

* :ref:`phonopyDispFileName`

* :ref:`phonopyBORNFileName`

* :ref:`qMesh`

* :ref:`temperatures`

* :ref:`minTemperature`

* :ref:`maxTemperature`

* :ref:`deltaTemperature`

* :ref:`smearingMethod`

* :ref:`smearingWidth`

* :ref:`constantRelaxationTime`

* :ref:`withIsotopeScattering`

* :ref:`masses`

* :ref:`isotopeCouplings`

* :ref:`boundaryLength`

* :ref:`deltaPath`

* :ref:`beginEndPointPath`


.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "phononLifetimes"
  phFC2FileName = "../Silicon/espresso.ifc2",
  sumRuleFC2 = "simple"
  phFC3FileName = "../Silicon/ShengBTEForceConstants3rd"
  qMesh = [15,15,15]
  temperatures = [600.]
  smearingMethod = "gaussian"
  smearingWidth = 10. cmm1
  deltaPath = 0.01
  begin point path
   L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
   G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
   X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
   K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
  end point path


-----------------------------------

Electron Lifetimes on a Path
-----------------------------

**Functionality:** Compute electron-phonon lifetimes/linewidths on a path through the Brillouin zone using Wannier interpolation.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronLifetimes"

* :ref:`phFC2FileName`

* :ref:`electronH0Name`

* :ref:`wsVecFileName`

* :ref:`sumRuleFC2`

* :ref:`elphFileName`

* :ref:`kMesh`

* :ref:`temperatures`

* :ref:`minTemperature`

* :ref:`maxTemperature`

* :ref:`deltaTemperature`

* :ref:`dopings`

* :ref:`chemicalPotentials`

* :ref:`minChemicalPotential`

* :ref:`maxChemicalPotential`

* :ref:`deltaChemicalPotential`

* :ref:`smearingMethod`

* :ref:`smearingWidth`

* :ref:`constantRelaxationTime`

* :ref:`numOccupiedStates`

* :ref:`fermiLevel`

* :ref:`deltaPath`

* :ref:`beginEndPointPath`


.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "electronLifetimes"
  phFC2FileName = "./silicon.fc",
  sumRuleFC2 = "crystal"
  electronH0Name = "./si_tb.dat",
  elphFileName = "./silicon.phoebe.elph.dat"
  kMesh = [15,15,15]
  temperatures = [600.]
  dopings = [1.e22]
  smearingMethod = "gaussian"
  smearingWidth = 0.5 eV
  deltaPath = 0.01
  begin point path
   L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
   G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
   X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
   K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
  end point path

-----------------------------------

Phonon Dos
----------

**Functionality:** Compute the phonon density of states.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "phononDos"

* :ref:`phFC2FileName`

* :ref:`phonopyDispFileName`

* :ref:`phonopyBORNFileName`

* :ref:`sumRuleFC2`

* :ref:`qMesh`

* :ref:`dosMinEnergy`

* :ref:`dosMaxEnergy`

* :ref:`dosDeltaEnergy`

* :ref:`masses`

.. raw:: html

  <h3>Sample input file</h3>

::

  phFC2FileName = "qespresso/silicon.fc",
  sumRuleFC2 = "simple"
  qMesh = [10,10,10]
  appName = "phononDos"
  dosMinEnergy = 0. cmm1
  dosMaxEnergy = 600. cmm1
  dosDeltaEnergy = 0.5 cmm1

--------------

Phonon Bands
------------

**Functionality:** Compute the phonon band structure on a path through the Brillouin zone.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "phononBands"

* :ref:`phFC2FileName`

* :ref:`phonopyDispFileName`

* :ref:`phonopyBORNFileName`

* :ref:`sumRuleFC2`

* :ref:`deltaPath`

* :ref:`beginEndPointPath`

* :ref:`masses`

* :ref:`outputEigendisplacements`

.. raw:: html

  <h3>Sample input file (Quantum ESPRESSO)</h3>

::

  appName = "phononBands"
  sumRuleFC2 = "simple"

  begin point path
   L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
   G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
   X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
   K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
  end point path

.. raw:: html

  <h3>Sample input file (phono3py)</h3>

::

  appName = "phononBands"
  phFC2FileName = "fc2.hdf5"
  phonopyDispFileName = "phono3py_disp.yaml"
  sumRuleFC2 = "simple"

  begin point path
   G 0.000 0.000 0.000  X 0.000 0.500 0.500
   X 0.000 0.500 0.500  W 0.250 0.750 0.500
   W 0.250 0.750 0.500  L 0.500 0.500 0.500
   L 0.500 0.500 0.500  G 0.000 0.000 0.000
   G 0.000 0.000 0.000  K 0.375 0.750 0.375
  end point path


-----------------------------------

Electron DoS
------------

Electron DoS (Wannier interpolation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Functionality:** Compute the electronic density of states. Electronic bands are interpolated to a finer mesh using Wannier interpolation.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronWannierDos"

* :ref:`electronH0Name`

* :ref:`wsVecFileName`

* :ref:`fermiLevel`

* :ref:`kMesh`

* :ref:`dosMinEnergy`

* :ref:`dosMaxEnergy`

* :ref:`dosDeltaEnergy`

* :ref:`beginEndCrystal`

.. raw:: html

  <h3>Sample input file</h3>

::

  electronH0Name = "qespresso/si_tb.dat",
  kMesh = [10,10,10]
  appName = "electronWannierDos"
  dosMinEnergy = -6. eV
  dosMaxEnergy = 20. eV
  dosDeltaEnergy = 0.1 eV
  begin crystal
   Si 0.00000   0.00000   0.00000
   Si 1.34940   1.34940   1.34940
  end crystal

-----------------------------------

Electron DoS (Fourier interpolation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Functionality:** Compute the electronic density of states. Electronic bands are interpolated to a finer mesh using Fourier interpolation.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronFourierDos"

* :ref:`electronH0Name`

* :ref:`kMesh`

* :ref:`fermiLevel`

* :ref:`dosMinEnergy`

* :ref:`dosMaxEnergy`

* :ref:`dosDeltaEnergy`

* :ref:`electronFourierCutoff`


.. raw:: html

  <h3>Sample input file</h3>

::

  electronH0Name = "qespresso/out/silicon.xml",
  kMesh = [10,10,10]
  appName = "electronFourierDos"
  dosMinEnergy = -6. eV
  dosMaxEnergy = 20. eV
  dosDeltaEnergy = 0.1 eV
  electronFourierCutoff = 4.

----------------------------------

Electron Bands
-----------------------------------

Electron Bands (Wannier interpolation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Functionality:** Compute the phonon band structure on a path through the Brillouin zone. Electronic bands are interpolated to a finer mesh using Wannier interpolation.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronWannierBands"

* :ref:`electronH0Name`

* :ref:`wsVecFileName`

* :ref:`fermiLevel`

* :ref:`deltaPath`

* :ref:`beginEndPointPath`

* :ref:`beginEndCrystal`


.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "electronWannierBands"
  electronH0Name = "qespresso/si_tb.dat",
  deltaPath = 0.01
  begin point path
   L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
   G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
   X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
   K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
  end point path
  begin crystal
   Si 0.00000   0.00000   0.00000
   Si 1.34940   1.34940   1.34940
  end crystal

-----------------------------------

Electron Bands (Fourier interpolation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Functionality:** Compute the electronic band structure on a path through the Brillouin zone. Electronic bands are interpolated to a finer mesh using Fourier interpolation.

.. raw:: html

  <h3>Input variables</h3>


:ref:`appName` = "electronFourierBands"

* :ref:`electronH0Name`

* :ref:`fermiLevel`

* :ref:`deltaPath`

* :ref:`electronFourierCutoff`

* :ref:`beginEndPointPath`

.. raw:: html

  <h3>Sample input file</h3>

::

  appName = "electronFourierBands"
  electronH0Name = "qespresso/out/silicon.xml",
  deltaPath = 0.01
  electronFourierCutoff = 4.
  begin point path
   L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
   G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
   X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
   K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
  end point path



-----------------------------------

Input Variable Descriptions
----------------------------

Below are descriptions of the input variables available in Phoebe, along with their formats and when applicable, default values. If no default value is listed, this means the default value is nothing. If a required variable is left blank, Phoebe will throw a related error message.

.. _appName:

appName
^^^^^^^

* **Description:** This parameter, which must always be present, identifies which app (functionality) you want to run.

* **Format:** *string*

* **Required:** yes

**Possible values:**
  * **"elPhQeToPhoebe":** app to convert electron-phonon coupling from QE to Phoebe format (must be run before running any electron transport).

  * **"phononTransport":** app to solve the phonon BTE and compute phonon transport properties.

  * **"electronWannierTransport":** app to solve the electron BTE and compute electron transport properties.

  * **"transportEPA":** app to solve the electron BTE and compute the electron transport properties using the EPA approximation.

  * **"phononLifetimes":** app to compute the phonon lifetimes on a given Brillouin zone path.

  * **"electronLifetimes":** app to compute the electron lifetimes on a given Brillouin zone path.

  * **"phononDos":** app to compute the phonon density of states.

  * **"phononBands":** app to compute the phonon bands on a Brillouin zone path.

  * **"electronWannierDos":** app to compute the electron density of states with Wannier interpolation.

  * **"electronFourierDos":** app to compute the electron density of states with Fourier interpolation.

  * **"electronWannierBands":** app to compute the electron bands with Wannier interpolation on a Brillouin zone path.

  * **"electronFourierBands":** app to compute the electron bands with Fourier interpolation on a Brillouin zone path.


.. _phFC2FileName:

phFC2FileName
^^^^^^^^^^^^^^

* **Description:** Path to a file containing harmonic force constants. File formats supported are: Quantum-ESPRESSO output of ``q2r.x`` (``prefix.fc``) or phono3py output (``fc2.hdf5``).

* **Format:** *string*

* **Required:** yes (for all phonon and electron-phonon apps)


.. _phFC3FileName:

phFC3FileName
^^^^^^^^^^^^^^

* **Description:** Path to a file containing anharmonic (3rd order) force constants. File formats supported are: ShengBTE (``FORCE_CONSTANTS_3RD``) or phono3py (``fc3.hdf5``).

* **Format:** *string*

* **Required:** yes (for phonon transport and lifetime apps)


.. _phonopyDispFileName:

phonopyDispFileName
^^^^^^^^^^^^^^^^^^^

* **Description:** Path to the ``phono3py_disp.yaml`` file output by phono3py. (In the case of running only the harmonic phonons with phonopy, this file is named ``phonopy_disp.yaml``).

* **Format:** *string*

* **Required:** yes (for calculations using phono3py)


.. _phonopyBORNFileName:

phonopyBORNFileName
^^^^^^^^^^^^^^^^^^^

* **Description:** Path to the ``BORN`` file in the `format as used by phonopy <https://phonopy.github.io/phonopy/input-files.html#born-optional>`_. This allows for the inclusion of the non-analytic correction to the IFC2s. For Phoebe, there is no need to worry about the unit conversion on the first line of this file. Most codes (VASP, QE) put these parameters in units of e. Please use these units.

* **Format:** *string*

* **Required:** no


.. _sumRuleFC2:

sumRuleFC2
^^^^^^^^^^

* **Description:** If specified, applies an acoustic sum rule to the phonon harmonic force constants. Allowed values are "simple" or "crystal", with the same algorithm and meaning of Quantum-ESPRESSO ``matdyn.x`` program.

* **Format:** *string*

* **Required:** yes (for phonon calculations)


.. _qMesh:

qMesh
^^^^^

* **Description:** Triplet of integers with the fine q-point Monkhorst-Pack mesh that will be used for Brillouin zone integrations of phonon properties.

* **Format:** *list of int*

* **Required:** yes (for phonon calculations)


.. _kMesh:

kMesh
^^^^^

* **Description:** Triplet of integers with the fine k-point Monkhorst-Pack mesh that will be used for Brillouin zone integrations of electronic properties. In electron-phonon transport calculations, `qMesh` is set to be equal to this value and does not need to be specified by the user.

* **Format:** *list of int*

* **Required:** yes (for electronic calculations)


.. _temperatures:

temperatures
^^^^^^^^^^^^

* **Description:** List with the values of temperatures (in Kelvin) to be used in the calculation. If scatteringMatrixInMemory=true, only one value of temperature is allowed.

* **Format:** *list of doubles*

* **Required:** yes (for transport and lifetime calculations)


.. _smearingMethod:

smearingMethod
^^^^^^^^^^^^^^

* **Description:** Selects the level of approximation for replacing the Dirac-delta approximating energy conservation. Allowed values are "gaussian" and "adaptiveGaussian" (preferred)

* **Format:** *string*

* **Required:** yes (for transport and lifetime calculations)


.. _smearingWidth:

smearingWidth
^^^^^^^^^^^^^

* **Description:** This parameter is required if :ref:`smearingMethod` = "gaussian", where this parameter represents the full-width half-maximum of the Gaussian used to approximate the Dirac-delta conserving energy. Example: smearingWidth = 0.5 eV

* **Format:** *double+units*

* **Required:** yes (when :ref:`smearingMethod` = "gaussian")


.. _solverBTE:

solverBTE
^^^^^^^^^

* **Description:** If specified, solves the Boltzmann equation beyond the relaxation time approximation. Allowed values are: "variational", "iterative", and "relaxons", see the Theory section for a detailed explanation. Example: solverBTE=["variational","relaxons"]

* **Format:** *list of strings*

* **Required:** no


.. _scatteringMatrixInMemory:

scatteringMatrixInMemory
^^^^^^^^^^^^^^^^^^^^^^^^

* **Description:** If true, the scattering matrix is kept in memory, and only one temperature is allowed. In exchange for a larger memory usage, exact BTE solvers are much faster. Disable this flag to reduce the memory footprint, at the cost of slowing down the exact BTE solvers.

* **Format:** *bool*

* **Required:** no

* **Default:** `true`



.. _symmetrizeMatrix:

symmetrizeMatrix
^^^^^^^^^^^^^^^^

* **Description:** If true, we enforce the symmetrix property of the scattering matrix A by doing A=(A^T+A)/2, where the transpose operation is made with respect to the wavevector indices. This operation increases the stability of the variational and relaxon solvers, but is computationally expensive in large system sizes. If you find your calculation has many negative relaxons eigenvalues, you might want to turn this on. This may also be favorable for your final production calculations.

* **Format:** *bool*

* **Required:** no

* **Default:** `false`


.. _numRelaxonsEigenvalues:

numRelaxonsEigenvalues
^^^^^^^^^^^^^^^^^^^^^^

* **Description:** Compute the relaxons solver using only the ``numRelaxonsEigenvalues`` largest eigenvalues + corresponding eigenvectors. This can dramatically reduce the cost of the calculation, as the largest eigenvalues comprise most of the result. However, you have to be careful to converge the calculation with respect to this parameter as well if you use it. It's great for testing your calculation, perhaps using ~25% of the eigenvalues, with your final production result using a full calculation.

Additionally, note that this leads to a second ScaLAPACK call to check for negative eigenvalues, which reduces the benefit of partial eigenvalue calculation. If you want to turn this off for additional cost reduction (though it's good to check this to ensure the quality of the scattering matrix) you can do so with :ref:`checkNegativeRelaxons` = false.

* **Format:** *integer*

* **Required:** no

* **Default:** `0` (this indicates the code should compute all eigenvalues)

.. _checkNegativeRelaxons:

checkNegativeRelaxons
^^^^^^^^^^^^^^^^^^^^^

* **Description:** When using the relaxons solver for only ``numRelaxonsEigenvalues`` largest relaxon eigenvalues, the check for negative eigenvalues (to ensure the quality of the calculation) is done by a second ScaLAPACK call. Thoguh it's good to inspect the output of this check, if you want to turn this off, set this variable to false for additional speedup.

* **Format:** *bool*

* **Required:** no

* **Default:** `true`


.. _distributedElPhCoupling:

distributedElPhCoupling
^^^^^^^^^^^^^^^^^^^^^^^

* **Description:** If true, the electron-phonon coupling in the Wannier representation is distributed over MPI processes in the QE to Phoebe application (the transform of the el-ph matrix elements to the real-space Wannier basis), helping reducing the memory requirements of a run. The MPI parallelization takes place over the number of irreducible q-points of the phonon calculation (which sets the upper number of MPI processes that can be used). If false, the electron-phonon coupling tensor is not distributed over MPI processes: calculations will be faster, but in exchange for a much larger memory requirement that can cause segmentation faults for some large use cases. The maximum possible distribution is num MPI processes = num dyn files. 

* **Format:** *bool*

* **Required:** no

* **Default:** `true`


.. _hdf5ElPhFileFormat:

hdf5ElPhFileFormat
^^^^^^^^^^^^^^^^^^

* **Description:** Use this parameter to change the format of the HDF5 file used to store the elcetron-phonon coupling. The default (1) should work for most cases. We found that the default file format may have issues for very large electron-phonon coupling tensors (>30Gb), due to possible overflows of the HDF5 library. If HDF5 displays problems, we suggest to either try to compile the code with the serial version of HDF5, or to set hdf5ElPhFileFormat to 2, to use a different format for the coupling tensor which circumvents some of the limitations of the HDF5 library.

* **Format:** *int*

* **Required:** no

* **Default:** `1`


.. _windowType:

windowType
^^^^^^^^^^

* **Description:** Enables the window used to discard phonon or electron states that don't contribute to transport. For phonon transport, we discard phonon states, and for electron transport, we discard electron states. Possible values are "nothing", "population" and "energy".
  * "nothing" means window is not applied.
  * "population" means phonon states are discarded if :math:`\frac{\partial \bar{n}}{\partial T} <` windowPopulationLimit, where :math:`\frac{\partial \bar{n}}{\partial T}` is the Bose--Einstein distribution derivative, with the same procedure used for electronic transport, just instead with a Fermi--Dirac function.
  * "energy" discards states which fall outside the :ref:`windowEnergyLimit`. States are removed at each wavevector point, which means each wavevector can have a different number of bands. Here, the user specifies with :ref:`windowEnergyLimit` the energy range desired using absolute energies, not those relative to the chemical potential.
  * "muCenteredEnergy" works almost identically to the "energy" window -- however, in this case, the user uses :ref:`windowEnergyLimit` to specify the energy range relative to the chemical potential. For example, if :math:`mu = 2` eV, and :ref:`windowEnergyLimit` = [-0.1, 0.1], only states with energies in the range [1.9, 2.1] eV would be included in the calculation.

* **Format:** *string*

* **Required:** no

* **Default:** `"nothing"`

.. _windowEnergyLimit:

windowEnergyLimit
^^^^^^^^^^^^^^^^^

* **Description:** Additional parameter for energy :ref:`windowType`. Specify two values :math:`E_{min}` and :math:`E_{max}` (in electronVolts) such that we discard all states (phonon or electron, depending on the calculation type) with energy outside of these bounds. When :ref:`windowType` = "muCenteredEnergy", this window specifies the energy range around the chemical potential to be included in the calculation (see :ref:`windowType` for more details).

* **Format:** *list of doubles*

* **Required:** yes (if :ref:`windowType` = "energy" or "muCenteredEnergy")


.. _windowPopulationLimit:

windowPopulationLimit
^^^^^^^^^^^^^^^^^^^^^

* **Description:** Used if :ref:`windowType` = "population". Cutoff values for discarding states (phonon or electron, depending on the calculation type) based on their equilibrium phonon occupation number, such that :math:`\frac{\partial \bar{n}}{\partial T} <` windowPopulationLimit.

* **Format:** *double*

* **Required:** no (optional if :ref:`windowType` = "population")

* **Default:** `1e-10`

.. _maxIterationsBTE:

maxIterationsBTE
^^^^^^^^^^^^^^^^

* **Description:** Maximum number of iterations for iterative and variational BTE solvers. If the maximum number of iterations is reached, the code will throw an error.

* **Format:** *int*

* **Required:** no

* **Default:** `50`


.. _convergenceThresholdBTE:

convergenceThresholdBTE
^^^^^^^^^^^^^^^^^^^^^^^

* **Description:** Convergence criterion to stop iterative BTE solvers. The calculation is converged if the transport coefficients have a relative change smaller than convergenceThresholdBTE.

* **Format:** *double*

* **Required:** no

* **Default:** `1.0e-5`


.. _dimensionality:

dimensionality
^^^^^^^^^^^^^^

* **Description:** Input the dimensionality of the material. As a result, transport coefficients tensors will be of size (dim x dim), and units will be suitably scaled for the desired dimensionality.

* **Format:** *int*

* **Required:** no

* **Default:** `3`


.. _constantRelaxationTime:

constantRelaxationTime
^^^^^^^^^^^^^^^^^^^^^^

* **Description:** If specified, we solve the BTE with the constant relaxation time approximation, where the electron or phonon lifetime is set to this input value. (Fast but inaccurate!)

* **Format:** *double+units*

* **Required:** no


.. _withIsotopeScattering:

withIsotopeScattering
^^^^^^^^^^^^^^^^^^^^^

* **Description:** Controls whether to include or not phonon-isotope scattering

* **Format:** *bool*

* **Required:** no

* **Default:** `true`


.. _masses:

masses
^^^^^^

* **Description:** User can specify a custom value of atomic masses. The masses must be ordered in the same way that atomic species are specified in the file phFC2FileName. If used, must specify the masses for all species. Defaults to the average mass for natural isotopic abundance.

* **Format:** *vector component*

* **Required:** no

* **Default:** average mass at natural isotopic abundance


.. _isotopeCouplings:

isotopeCouplings
^^^^^^^^^^^^^^^^

* **Description:** User can specify a list of custom atomic mass isotopic coupling parameters :math:`g_2^s`. See Theory section for a description. The values of isotopic scattering must be ordered in the same way that atomic species are specified in the file phFC2FileName. If used, must specify the couplings for all species. Defaults to the mass isotope scattering for natural isotopic abundance.

* **Format:** *vector component*

* **Required:** no

* **Default:** natural isotopic abundance


.. _boundaryLength:

boundaryLength
^^^^^^^^^^^^^^

* **Description:** If specified, includes the phonon-boundary scattering within the RTA approximation. Example: boundaryLength = 10 mum

* **Format:** *double+units*

* **Required:** no


.. _electronH0Name:

electronH0Name
^^^^^^^^^^^^^^

* **Description:** For Wannier-interpolation-based calculations, electronH0Name must contain the path to the ``{prefix}_tb.dat`` file generated by Wannier90. For Fourier-interpolation-based calculations, electronH0Name must contain the path to the Quantum-ESPRESSO ``{outdir}/{prefix}.xml`` file generated by ``pw.x``.

* **Format:** *string*

* **Required:** yes


.. _wsVecFileName:

wsVecFileName
^^^^^^^^^^^^^

* **Description:** The file ``*_wsvec.dat`` generated by Wannier90 contains additional information to make the Wannier-interpolation of electronic bands more accurate. Specifically, the file contains Wannier-function dependent shifts on the phase factors. See the theory section for more details. If this file is linked, the code will interpolate the band structure using the information of this file. If not, the code will ignore the phase-factor shifts.

* **Format:** *string*

* **Required:** no


.. _dosMinEnergy:

dosMinEnergy
^^^^^^^^^^^^

* **Description:** Used in conjunction with :ref:`dosMaxEnergy` and :ref:`dosDeltaEnergy` to compute the Density of States every :ref:`dosDeltaEnergy` increments between :ref:`dosMinEnergy` and :ref:`dosMaxEnergy`.

* **Format:** *double+units*

* **Required:** yes (for dos calculation)


.. _dosMaxEnergy:

dosMaxEnergy
^^^^^^^^^^^^

* **Description:** Used in conjunction with :ref:`dosMinEnergy` and :ref:`dosDeltaEnergy` to compute the Density of States every :ref:`dosDeltaEnergy` increments between :ref:`dosMinEnergy` and :ref:`dosMaxEnergy`.

* **Format:** *double+units*

* **Required:** yes (for dos calculation)


.. _dosDeltaEnergy:

dosDeltaEnergy
^^^^^^^^^^^^^^

* **Description:** Used in conjunction with :ref:`dosMinEnergy` and :ref:`dosMaxEnergy` to compute the Density of States every :ref:`dosDeltaEnergy` increments between :ref:`dosMinEnergy` and :ref:`dosMaxEnergy`.

* **Format:** *double+units*

* **Required:** yes (for dos calculation)


.. _electronFourierCutoff:

electronFourierCutoff
^^^^^^^^^^^^^^^^^^^^^

* **Description:** A parameter controlling the search of lattice vectors used for the Fourier interpolation of the electronic band structure. In detail, the lattice vectors used for the Fourier interpolation are searched in a supercell of size `electronFourierCutoff`:sup:`3` the primitive unit cell. Set it to at least 2.

* **Format:** *double*

* **Required:** yes (for electron Fourier DoS and bands)


.. _beginEndCrystal:

begin/end crystal
^^^^^^^^^^^^^^^^^

* **Description:** Specify the atomic species and atomic positions inside the crystal. This needs to be specified in some apps like WannierBands or WannierDos, as the output files of Wannier90 doesn't provide all the information about the crystal.

* **Format:** Namelist format, with atomic symbol and position coordinates in units of Angstroms. Example::

    begin crystal
     Si 0.00000   0.00000   0.00000
     Si 1.34940   1.34940   1.34940
    end crystal

* **Required:** yes (for electron Wannier DoS and bands calculations)


.. _beginEndPointPath:

begin/end point path
^^^^^^^^^^^^^^^^^^^^

* **Description:** Specify the path of wavectors in the Brillouin zone used in apps such as `phononBands` or `phononLifetimes`. Use the parameter :ref:`deltaPath` to control the number of wavevectors in each segment.

* **Format:** Namelist format, as pairs of special point symbol and wavevector coordinates. Wavevector coordinates are in fractional coordinates with respect to the primitive reciprocal lattice vectors. Example::

    begin point path
     L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
     G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
     X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
     K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
    end point path

* **Required:** yes (for electron and phonon bands and lifetime apps)

.. _dopings:

dopings
^^^^^^^

* **Description:** Specify a list of doping concentrations, in cm :sup:`-3`, to compute electronic properties at various doping concentrations. The chemical potentials corresponding to this doping concentrations will be computed.
Phoebe uses the convention that a negative value corresponds to n-type doping (moves chemical potential towards conduction bands) and a positive value corresponds to p-type doping (moves chemical potential downwards into valence bands).

* **Format:** *list of doubles*

* **Required:** yes (for electron transport and lifetime calculations, unless :ref:`chemicalPotentials` is specified)


.. _chemicalPotentials:

chemicalPotentials
^^^^^^^^^^^^^^^^^^

* **Description:** Specify a list of chemical potentials to be used for the calculation of properties as a function of the chemical potential. If used in electron Wannier transport and scatteringMatrixInMemory=true, then only one value of chemical potentials can be specified. Values are in eV.

* **Format:** *list of doubles*

* **Required:** yes (unless :ref:`minChemicalPotential`, :ref:`maxChemicalPotential`, :ref:`deltaChemicalPotential` variables are present, or :ref:`dopings` are specified).


.. _elphFileName:

elphFileName
^^^^^^^^^^^^

* **Description:** Path to the file generated by the app `elPhQeToPhoebe` containing the electron-phonon coupling in the Wannier representation (e.g. ``{prefix}.phoebe.elph.dat``)

* **Format:** *string*

* **Required:** yes (for electron transport and lifetime apps)


.. _epaMinEnergy:

epaMinEnergy
^^^^^^^^^^^^

* **Description:** Specifies the minimum the energy bin value over which the electron-phonon coupling will be averaged for the EPA approximation post-processing in the ``elPhQeToPhoebe`` app.

* **Format:** *double*

* **Required:** yes (for EPA ``elPhQeToPhoebe`` runs)


.. _epaMaxEnergy:

epaMaxEnergy
^^^^^^^^^^^^

* **Description:** Specifies the maximum the energy bin value over which the electron-phonon coupling will be averaged for the EPA approximation post-processing in the ``elPhQeToPhoebe`` app.

* **Format:** *double*

* **Required:** yes (for EPA ``elPhQeToPhoebe`` runs)


.. _epaNumBins:

epaNumBins
^^^^^^^^^^^

* **Description:** The number of energy bins, ranging from :ref:`epaMinEnergy` to :ref:`epaMaxEnergy` used to average the electron-phonon matrix elements for EPA calculations in the ``elPhQeToPhoebe`` app.

* **Format:** *int*

* **Required:** yes (for EPA ``elPhQeToPhoebe`` runs)


.. _epaSmearingEnergy:

epaSmearingEnergy
^^^^^^^^^^^^^^^^^

* **Description:** Specifies the Gaussian width used in the moving least squares averaging procedure used to average the electron-phonon matrix elements for an EPA calculation.

* **Format:** *double*

* **Required:** yes (for EPA ``elPhQeToPhoebe`` runs)


.. _epaFileName:

epaFileName
^^^^^^^^^^^

* **Description:** This is the path to the file ``*.phoebe.epa.dat``, which is created by ``elPhQeToPhoebe``.

* **Format:** *string*

* **Required:** yes (for EPA transport app)


.. _epaEnergyStep:

epaEnergyStep
^^^^^^^^^^^^^

* **Description:** The energy interval used to integrate the transport coefficients, i.e. lifetimes will be computed every ``epaEnergyStep`` energies.

* **Format:** *double*

* **Required:** yes (for EPA transport app)


.. _epaEnergyRange:

epaEnergyRange
^^^^^^^^^^^^^^

* **Description:** EPA lifetimes will be computed for all energies in proximity of the chemical potential, i.e. for all energies such that :math:`|\epsilon-\mu|<\text{epaEnergyRange}`. This variable specifies that range.

* **Format:** *double*

* **Required:** yes (for EPA transport app)


.. _deltaPath:

deltaPath
^^^^^^^^^

* **Description:** This variable controls how far apart are the wavevectors when a path in the Brillouin zone is specified, and it represents the distance between wavevectors on the band path in crystal coordinates. Can be used when a path of wavevectors is specified with the :ref:`beginEndPointPath` key.

* Default: 0.05

* **Format:** *string*

* **Required:** no


.. _outputEigendisplacements:

outputEigendisplacements
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Description:** Optional variable which outputs the phonon eigendisplacements to the ``phonon_bands.json`` file when using the ``phononBands`` app. See the :ref:`eigendisplacements` tutorial for more information on use.

* **Default:** false

* **Format:** *bool*

* **Required:** no

.. _elPhInterpolation:

elPhInterpolation
^^^^^^^^^^^^^^^^^

* **Description:** Can be either "wannier" or "epa". The first prepares the electron-phonon coupling for the transport calculation with Wannier interpolation (i.e. does the transformation from Bloch to Wannier representation). The second, prepares the electron-phonon coupling to be used with the EPA approximation.

* **Format:** *string*

* **Required:** yes (for elPhQeToPhoebe app)


.. _wannier90Prefix:

wannier90Prefix
^^^^^^^^^^^^^^^

* **Description:** Set to the same value of ``prefix`` in Wannier90. It's used to locate the files ``{prefix}.eig`` generated by ``wannier90.x``.

* **Format:** *string*

* **Required:** yes (for any electron app using Wannier interpolation)


.. _quantumEspressoPrefix:

quantumEspressoPrefix
^^^^^^^^^^^^^^^^^^^^^

* **Description:** Set to the same value of ``prefix`` in Quantum-ESPRESSO. It's used to locate the files ``{prefix}.dyn*`` or ``{prefix}.phoebe.*.dat`` generated by ``ph.x``.

* **Format:** *string*

* **Required:** yes


.. _fermiLevel:

fermiLevel
^^^^^^^^^^

* **Description:** Sets the fermi level of the ground state. Can be specified e.g. in Bands or DOS apps to specify an otherwise unknown fermi level. This quantity is read from file for transport calculations: this input parameter overwrites that value, use with caution.

* **Format:** *double+units*

* **Required:** no


.. _numOccupiedStates:

numOccupiedStates
^^^^^^^^^^^^^^^^^

* **Description:** Determines the number of occupied Kohn-Sham states at the ground state. The default value might be read from the :ref:`electronH0Name` (when this is the Quantum-ESPRESSO xml file) or the file with the el-ph interaction (so, the user may not need to specify it for transport calculations). This value controls where the Fermi level is set. The user alternatively can specify the :ref:`fermiLevel` (and :ref:`numOccupiedStates` will be computed from the Fermi level).

* **Format:** *double*

* **Required:** no


.. _minChemicalPotential:

minChemicalPotential
^^^^^^^^^^^^^^^^^^^^

* **Description:** To be used together with :ref:`maxChemicalPotential` and :ref:`deltaChemicalPotential`, sets the code to compute properties at all chemical Potentials between :ref:`minChemicalPotential` and :ref:`maxChemicalPotential` in steps of :ref:`deltaChemicalPotential`. Can be exchanged with :ref:`chemicalPotentials` to instead manually specify the chemical potentials of the calculation.

* **Format:** *double*

* **Required:** yes (either specify :ref:`minChemicalPotential`, :ref:`maxChemicalPotential`, and :ref:`deltaChemicalPotential` **or** :ref:`chemicalPotentials`.


.. _maxChemicalPotential:

maxChemicalPotential
^^^^^^^^^^^^^^^^^^^^

* **Description:** To be used together with :ref:`minChemicalPotential` and :ref:`deltaChemicalPotential`, sets the code to compute properties at all chemical potentials between :ref:`minChemicalPotential` and :ref:`maxChemicalPotential` in steps of :ref:`deltaChemicalPotential`. Can be exchanged with chemicalPotentials to instead manually specify the chemical potentials of the calculation.

* **Format:** *double*

* **Required:** yes (either specify :ref:`minChemicalPotential`, :ref:`maxChemicalPotential`, and :ref:`deltaChemicalPotential` **or** :ref:`chemicalPotentials`)


.. _deltaChemicalPotential:

deltaChemicalPotential
^^^^^^^^^^^^^^^^^^^^^^

* **Description:** To be used together with :ref:`minChemicalPotential` and :ref:`maxChemicalPotential`, sets the code to compute properties at all chemical Potentials between :ref:`minChemicalPotential` and :ref:`maxChemicalPotential` in steps of :ref:`deltaChemicalPotential`. Can be exchanged with :ref:`chemicalPotentials` to instead manually specify the chemical potentials of the calculation.

* **Format:** *double*

* **Required:** yes (either specify :ref:`minChemicalPotential`, :ref:`maxChemicalPotential`, and :ref:`deltaChemicalPotential` **or** :ref:`chemicalPotentials`)


.. _minTemperature:

minTemperature
^^^^^^^^^^^^^^

* **Description:** To be used together with :ref:`maxTemperature` and :ref:`deltaTemperature`, sets the code to compute observables at temperatures (in Kelvin) between :ref:`minTemperature` and :ref:`maxTemperature` in steps of :ref:`deltaTemperature`.

* **Format:** *double*

* **Required:** yes (either set :ref:`minTemperature`, :ref:`maxTemperature`, and :ref:`deltaTemperature` **or** :ref:`temperatures`)


.. _maxTemperature:

maxTemperature
^^^^^^^^^^^^^^

* **Description:** To be used together with :ref:`minTemperature` and :ref:`deltaTemperature`, sets the code to compute observables at temperatures (in Kelvin) between :ref:`minTemperature` and :ref:`maxTemperature` in steps of :ref:`deltaTemperature`.

* **Format:** *double*

* **Required:** yes (either set :ref:`minTemperature`, :ref:`maxTemperature`, and :ref:`deltaTemperature` **or** :ref:`temperatures`)


.. _deltaTemperature:

deltaTemperature
^^^^^^^^^^^^^^^^

* **Description:** To be used together with minTemperature and maxTemperature, sets the code to compute observables at temperatures (in Kelvin) between :ref:`minTemperature` and :ref:`maxTemperature` in steps of :ref:`deltaTemperature`.

* **Format:** *double*

* **Required:** yes (either set :ref:`minTemperature`, :ref:`maxTemperature`, and :ref:`deltaTemperature` **or** :ref:`temperatures`)


.. _useSymmetries:

useSymmetries
^^^^^^^^^^^^^

* **Description:** When set to true, triggers the BTE to be computed only on the irreducible wedge of the Brillouin zone. For systems with several symmetries, this speeds up calculations. On the other hand, it may slow down the code for systems with few symmetries. If set to true, the viscosity is only computed at the RTA level.

* **Format:** *bool*

* **Required:** no

* **Default:** `false`

