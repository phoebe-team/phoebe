Electron-Phonon Averaged (EPA) Transport Tutorial
=================================================

Synopsis
--------

In this tutorial, we will compute the electrical conductivity and other electronic transport properties of silicon using the electron-phonon averaged approximation (EPA).

We will use Quantum ESPRESSO to compute the ab-initio electron-phonon coupling on a coarse grid, convert the output to an input for Phoebe, and then perform the EPA approximation to inexpensively calculate electronic transport properties as detailed in `j.mtphys.2018.07.001 <https://doi.org/10.1016/j.mtphys.2018.07.001>`__, which is covered in the :ref:`theoryEPA` section of the theory documentation.

We assume a basic knowledge of Quantum ESPRESSO (see tutorials on phonon calculations on `Quantum ESPRESSO's website <https://www.quantum-espresso.org/resources/tutorials>`__).


Step 1: Patch Quantum ESPRESSO
------------------------------
We need to use a custom modification of Quantum ESPRESSO (which we modified to impose the symmetric properties of the wavefunction).
To do this, we must compile a new copy of Quantum ESPRESSO.

From an installation folder of your choice, type::

    # download the patched version
    git clone https://github.com/mir-group/phoebe-quantum-espresso.git
    cd phoebe-quantum-espresso
    # install it
    git checkout phoebe-qe-7.0
    ./configure MPIF90=mpif90 --with-scalapack=yes
    make pw pp ph w90

where you should modify the file paths and ``./configure`` arguments for QE to match your system.
If compilation fails, you should consult the QE `installation page <https://www.quantum-espresso.org/Doc/user_guide/node7.html>`__.



Step 2: Run pw.x
-----------------

First, we need to compute the total energy of the silicon unit cell.
This calculation will create the ground state charge density and wavefunctions that are needed for later.

To run this calculation, go to the folder ``./example/Silicon_epa/qe-elph`` in the Phoebe repository.
The file ``scf.in`` is the input file for the ``pw.x`` executable.
The contents of the ``scf.in`` file for a total energy DFT calculation of Quantum ESPRESSO for a silicon crystal is::

        &control
          calculation = "scf"
          prefix = "silicon"
          pseudo_dir = "../../pseudoPotentials/"
          outdir = "./out"
          verbosity = "high"
          wf_collect = .true.
        /
        &system
          ibrav = 2
          celldm(1) = 10.2
          nat = 2
          ntyp = 1
          ecutwfc = 30.0
          nbnd = 12
        /
        &electrons
          conv_thr = 1.0d-14
        /
        ATOMIC_SPECIES
          Si  28.086  Si.pz-vbc.UPF
        ATOMIC_POSITIONS alat
          Si 0.00 0.00 0.00
          Si 0.25 0.25 0.25
        K_POINTS automatic
        6 6 6 0 0 0

A detailed description of these parameters can be found on `Quantum ESPRESSO's website <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`__.
The most important parameters, which should be tweaked and modified in a research project are:

* **nbnd:** the number of Kohn-Sham states (bands) to be computed.

* **K_POINTS:** the parameter controlling the integration mesh of wavevectors on the Brillouin zone. Phonon properties should be converged against this mesh (more wavevectors is better). Tip: ``ph.x`` calculations are faster when the k-mesh is gamma-centered.

* **ecutwfc:** the parameter controlling the number of G-vectors used in the plane-wave expansion of the wavefunction. Phonon frequencies should be converged against this value.

* **conv_thr:** this parameter controls the total energy convergence threshold. Note that a low value of conv_thr may result in poorly converged phonon properties.

* **prefix:** prefix of some output files. Make sure to use a consistent value of prefix throughout your calculations.

* **outdir**: name of the scratch folder. Must be used consistently throughout the run so that it points to the correct files.

This list is obviously not complete, and for your research project you may need to use more functionalities from QE's ``pw.x``.

This is a quick calculation, but if there are a few cores available, you could parallelize it as::

    mpirun -np 4 /path/to/patched-quantum-espresso/bin/pw.x -npool 4 -in scf.in > scf.out

where here we have acted as if we have 4 cores. Be sure to substitute in the suitable path to the ``pw.x`` executable.

.. note::
   The patched QE used for Phoebe only supports the keyword ``K_POINTS automatic``.


Step 3: Phonons and electron-phonon coupling
--------------------------------------------

Next, we use the ``ph.x`` executable from our patched QE to run a phonon calculation, during which the electron-phonon matrix elements on a coarse mesh are computed. The input file ``ph.in`` is as follows::

 phonons of Si
 &inputph
  tr2_ph = 1.0d-14
  prefix = "silicon"
  ldisp = .true.
  nq1 = 6
  nq2 = 6
  nq3 = 6
  outdir = "./out"
  fildyn = "silicon.dyn"
  fildvscf = "silicon.dvscf"
  electron_phonon = "epa"
 /

The values of ``nqX`` select the Monkhorst-Pack grid of q-points centered at Gamma, for which we will compute the phonon properties.
Also, it's important that ``prefix`` and ``outdir`` are the same as those used in the ``pw.x`` calculation from step 2.
Use a good value of ``tr2_ph`` (smaller is better, but harder to converge), which (indirectly) checks the convergence of phonon frequencies.

In the input file, we set the flag ``electron_phonon = "epa"``.
This will trigger the calculation of the electron-phonon coupling matrix elements which are used by Phoebe.

Run the code as::

  /path/to/patched-quantum-espresso/bin/ph.x -in ph.in > ph.out

Or in parallel, e.g.::

  mpirun -np 4 /path/to/patched-quantum-espresso/bin/ph.x -npool 4 -in ph.in > ph.out

If the code executes correctly and completely, you should see a number of files called ``{fildyn}*``, as many files as the number of irreducible q-points (16 in this case).
Additionally, you should also see several files named ``{prefix}.phoebe.****.dat``, as many as the number of irreducible points.
These files contain the electron-phonon coupling matrix elements to be used by Phoebe.

**Current limitations:**

* There are restrictions to the choice of k and q points.
  The ``K_POINTS`` in ``pw.x`` must be ``automatic``. The ``K_POINTS`` must be gamma centered.
  And the q-point mesh must be the same as the k-point mesh.

* In the current release, we don't support spin-polarized calculations or spin-orbit calculations. Support for this will come in a later release (we need to implement spin-related symmetries).


Step 4: Run q2r.x
-----------------

``ph.x`` has created a set of ``silicon.dyn*`` files, which contain the dynamical matrix at every irreducible q-point.
Now, we run ``q2r.x`` in order to Fourier transform the dynamical matrices in the reciprocal space representation to the real space representation, where they represent the harmonic interatomic force constants.
The input file ``q2r.in`` is minimal::

 &input
   fildyn='silicon.dyn',
   flfrc='silicon.fc'
 /

where the first variable must match the path to the dynamical matrices set earlier in ``ph.x``, and ``flfrc`` is the output file with the force constants.

In the working folder ``./example/Silicon-epa/qe-elph`` run the command::

    /path/to/patched-quantum-espresso/bin/q2r.x -in q2r.in > q2r.out

If the code run successfully, you should see a new file ``silicon.fc``.


Step 5: Non-self-consistent run
-------------------------------

Before we can run Phoebe, we need to complete one more step using Quantum ESPRESSO. We need to use an non-self-consistent run to calculate the electronic properties on the k-point mesh.
We do so using the input file ``bands.in`` in the ``Silicon-epa`` example folder::

  &control
    calculation = "bands"
    restart_mode = "from_scratch"
    prefix = "silicon"
    pseudo_dir = "../../pseudoPotentials/"
    outdir = "./out"
  /
  &system
    ibrav = 2
    celldm(1) = 10.2
    nat = 2
    ntyp = 1
    ecutwfc = 30.
    nbnd = 12
  /
  &electrons
    conv_thr = 1.0d-10
  /
  ATOMIC_SPECIES
    Si  28.086  Si.pz-vbc.UPF
  ATOMIC_POSITIONS alat
    Si 0.00 0.00 0.00
    Si 0.25 0.25 0.25
  K_POINTS crystal
  216
    0.00000000  0.00000000  0.00000000  4.629630e-03
    0.00000000  0.00000000  0.16666667  4.629630e-03
    ...

where the k-points list will continue for all 216 points. To generate this k-point list, one could use the ``kmesh.pl`` utility from Wannier90 (in the directory ``q-e/wannier90-3.0.0/utility/kmesh.pl``, used as ``kmesh.pl nk1 nk2 nk3``, with the output appended to the end of ``bands.in``).

.. note::
   The ``calculation`` parameter should be set to ``bands`` and not ``nscf``.

We run this as we did the ``pw.x`` step::

    mpirun -np 4 /path/to/patched-quantum-espresso/bin/pw.x -npool 4 -in bands.in > bands.out

where again this could be parallelized using ``mpi`` and ``npool``.


Step 6: QE to Phoebe conversion
-------------------------------

Now that we have generated all the necessary input files, we can get started with Phoebe.
In this section, we read all the information from the files created above and use them to prepare the electron-phonon coupling for the transport calculation.

In the case of an EPA calculation, this means transforming the electron-phonon coupling to the proper representation.
To do this, let's have a look at the input file ``qeToPhoebeEPA.in``::

  appName = "elPhQeToPhoebe"
  elPhInterpolation = "epa"

  phFC2FileName = "qe-elph/silicon.fc"
  electronH0Name = "qe-elph/out/silicon.xml",
  quantumEspressoPrefix = "qe-elph/silicon"

  electronFourierCutoff = 4.
  epaMinEnergy = -4. eV
  epaMaxEnergy = 10. eV
  epaNumBins = 40
  epaSmearingEnergy = 0.05 eV

The parameters in this input file are as follows:

* :ref:`appName` = `"elPhQeToPhoebe"`: here, we select the app to post-process the electron-phonon coupling created by QE.

* :ref:`elPhInterpolation` = `"epa"`: this selects the post-processing method. In this case, we choose the mode which transforms the electron-phonon coupling to the EPA representation.

* :ref:`phFC2FileName` = `"silicon.fc"`: this chooses the path to the phonon dynamical matrix.

* :ref:`electronH0Name`: points to the Quantum-ESPRESSO ``*.xml`` file created by ``pw.x``, which contains the electronic single-particle energies.

* :ref:`quantumEspressoPrefix` = `"silicon"`: this parameter is used to locate and read the files ``./silicon.phoebe.*.dat`` that have been created by ``ph.x``. You should set it to the ``prefix`` variable you chose when running QE in earlier steps.

* :ref:`electronFourierCutoff` = 4: this is a parameter used to control the Fourier interpolation of the electronic band structure. In this case, 4 implies that we will use all Bravais lattice vectors over a supercell of 4x4x4 times larger than the input unit cell.

* :ref:`epaMinEnergy`, :ref:`epaMaxEnergy`, :ref:`epaNumBins`: these last three parameters identify the values of energy (from min to max with numBins values) over which the electron-phonon coupling will be averaged.

* :ref:`epaSmearingEnergy`: is the Gaussian width used in the moving least squares averaging procedure.

The last 4 parameters are parameters which will determine the quality of the EPA calculation, and should be adjusted by the user for a production calculation.
Energies should cover the area around the Fermi level or HOMO and LUMO (which can be found in the output of ``pw.x``),
and the value of ``epaSmearingEnergy`` should be comparable to the size of the energy bin for the el-ph coupling.
As a suggestion, we also tend to find that not many energy bins are needed for this averaging procedure, as the el-ph coupling tends to be slowly varying with energy.

To execute the code::

  export OMP_NUM_THREADS=4
  mpirun -np 1 /path/to/phoebe/build/phoebe -in qeToPhoebeEPA.in -out qeToPhoebeEPA.out

and wait until completion.

Note that this calculation can be memory intensive.
For this reason, we recommend to limit/avoid use of MPI parallelization and use a large number of OMP threads (if you compiled the code with OpenMP. OpenMP is useful, because it allows multiple threads to work on a problem while sharing the memory on a node).
For some large calculations, the electron-phonon coupling tensor may be very large, so that a single MPI process cannot store an entire copy of the tensor in its own memory.
If this is the case (e.g. if some segmentation faults appear), you can try setting the input variable :ref:`distributedElPhCoupling` = `"true"`: this will decrease the memory requirements of the calculation in exchange for a slower calculation, and will parallelize with MPI over the irreducible q-points.

After the code completes, you should see an output file called ``silicon.phoebe.epa.dat``.


Step 7: EPA Electronic Transport
--------------------------------

Finally, you reached the last step, and now we can see some transport properties!
Below is an example input file for computing electronic transport properties::

  appName = "transportEpa"

  useSymmetries = true
  electronH0Name = "qe-elph/out/silicon.xml",
  epaFileName = "qe-elph/silicon.phoebe.epa.dat"

  electronFourierCutoff = 4.
  epaEnergyStep = 0.01 eV
  epaEnergyRange = 3.0 eV

  kMesh = [10,10,10]
  temperatures = [300.]
  dopings = [1.0e21]

The parameters used here are:

* :ref:`appName` = `"transportEPA"`: selects the app for computing electronic transport properties with EPA.

* :ref:`useSymmetries`: whether or not to use symmetries of the crystal structure in the calculation. 

* :ref:`electronH0Name`: points to the Quantum-ESPRESSO ``*.xml`` file created by ``pw.x``, which contains the electronic single-particle energies.

* :ref:`epaFileName`: is the path to the file created at the previous step with ``elPhQeToPhoebe``.

* :ref:`electronFourierCutoff`: as done above, this value controls the quality of the Fourier interpolation of the band structure, and, here, is set to interpolate using the Bravais lattice vector of a 4x4x4 supercell.

* :ref:`epaEnergyStep`: is the energy interval used to integrate the transport coefficients, i.e. lifetimes will be computed every ``epaEnergyStep`` energies.

* :ref:`epaEnergyRange`: lifetimes will be computed for all energies in proximity of the chemical potential, i.e. for all energies such that :math:`|\epsilon-\mu|<\text{epaEnergyRange}`.

* :ref:`kMesh`: is the grid used to integrate the Brillouin zone for obtaining the density of states.

* :ref:`temperatures` a list of temperatures in Kelvin for which we will compute transport properties.

* :ref:`dopings`: in cm :sup:`-3` at which we will compute results. This is only meaningful for semiconductors.


To run the code, we can simply do::

  export OMP_NUM_THREADS=4
  mpirun -np 1 /path/to/phoebe/build/phoebe -in epaTransport.in -out epaTransport.out


Note that the most time-consuming step of this calculation typically is the calculation of the density of states.
However, this is still dramatically faster than a Wannier-based transport technique.

Output
------

As usual, there are two kinds of output: the standard output file (in the line above, it's ``epaTransport.out``) and the JSON files containing more extensive transport and lifetime values.

.. raw:: html

  <h4>Standard Output File</h4>

This file shows results as well as a report of the calculation progress. The transport coefficients will print to the standard output file alongside with information on the chemical potential/doping and temperature used.

.. raw:: html

  <h4>JSON Output Files</h4>

There are several JSON files containing all the output, including the transport properties. They also contain information about the dos and electron bandstrucutre, as well as the units associated which each kind of output. It's worth opening and printing the keys from each JSON file to see the information in each file.

You can learn more about how to post-process these files at :ref:`postprocessing`.

**Files which are always output for this calculation:**

* ``epa_dos.json``: contains the electron density of states used in the calculation.

* ``epa_onsager_coefficients.json``: contains the electronic transport coefficients from EPA.

* ``epa_relaxation_times.json``: contains the EPA relaxation times at each energy bin value.


To understand how to parse these files in more detail, take a look at the scripts described by the :ref:`postprocessing` page. In particular, if you want to plot lifetimes vs. energy, look at ``epa_tau.py``. If you want to plot the transport coefficients vs. doping or temperature, check out ``transport_coefficients.py``.

.. note::
   It's a good idea to also use bands.py to plot the band structure. Fourier interpolation of the band structure can be a source of error -- you may need to go to higher k-point meshes to get a reasonable interpolation of the band structure. Check the bands at this stage to make sure they are similar to the true DFT bands -- otherwise, your results could be problematic!

Convergence Checklist
----------------------

In this tutorial, we show a demo calculation, which is certainly unconverged. We don't discuss the convergence tests that need to be done for a production/publication quality research project.

**You should make sure to test the convergence of:**

* Check that the phonon frequencies are converged with respect to k-point sampling, q-point sampling and wavefunction cutoff.

* Test that the electronic bandstructure is converged with respect to the k-point sampling, the ``ecutwfc`` (and ``ecutrho``) parameters of ``pw.x`` as well as the interpolating cutoff ``electronFourierCutoff``.

* Test the convergence of the electronic transport coefficients with respect to ab-initio results, in particular with respect to the k/q-point sampling in the DFT calculation.

* Check the convergence of the electronic transport results with respect to the energy bins used in the EPA approximation

* Test the convergence of the density of states w.r.t. the ``kMesh`` parameter.

Parallelization
---------------

* **For the qeToPhoebeWannier app:** The electron-phonon coupling tensor may be very large, so that a single MPI process cannot store an entire copy of the tensor in its own memory.

  If this is the case, you can try setting the input variable :ref:`distributedElPhCoupling` = `"true"`: this will decrease the memory requirements of the calculation in exchange for a slower calculation, and will parallelize with MPI over the irreducible q-points.


