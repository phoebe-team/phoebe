.. _elWanTransport:

Electron Wannier Transport Tutorial
===================================

Synopsis
--------

In this tutorial, we will compute the electrical conductivity and other electronic transport properties of silicon using the Wannier interpolation method.

First, we will use Quantum ESPRESSO to compute the ab-initio electron-phonon coupling on a coarse grid.
Then, we will use Wannier90 to compute the maximally localized Wannier functions, which we need to interpolate the electronic band structure and the electron-phonon matrix elements to a fine mesh. Steps 1-4 are quite similar to the equivalent steps from the previous tutorial.

The algorithms are described in the :ref:`theory` section of this manual. Again, we assume a basic familiarity with Quantum ESPRESSO (see phonon tutorials on `Quantum ESPRESSO's website <https://www.quantum-espresso.org/resources/tutorials>`__), and a working knowledge of Wannier90, although you should still be able to follow this tutorial.


Step 1: Patch Quantum ESPRESSO
------------------------------
As in the previous tutorial, we need to use a custom version of Quantum ESPRESSO (which we modified to impose the symmetric properties of the wavefunction).
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

Again, we need to compute the total energy of the silicon unit cell.
This calculation will create the ground state charge density and wavefunctions that are needed for later. This very similar to the second step of the EPA tutorial.

.. note::
   Very important! In this step, we are fixing the gauge of the wavefunction.
   It is imperative that this `scf` calculation is only done once at the beginning and is never repeated aftwerwards.
   If you run another `scf` calculation between step 3 and 6, you may alter the gauge of the wavefunction and thus ruin the interpolation of the electron-phonon coupling.

   Additionally, please make sure you do all steps involving QE (3-6) in the same directory with this gauge information -- otherwise, the calculation could be wrong!

To run this calculation, go to the folder ``./example/Silicon_el/qe-elph`` in the Phoebe repository.
The file ``scf.in`` is the input file for the ``pw.x`` executable.
The contents of the ``scf.in`` file for a total energy DFT calculation of Quantum ESPRESSO for a silicon crystal is shown below ::

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
  ATOMIC_POSITIONS crystal
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

Simply run it as::

    /path/to/patched-quantum-espresso/bin/pw.x -in scf.in > scf.out

after substituting the suitable path to the ``pw.x`` executable.

.. note::
   The patched QE used for Phoebe only supports the keyword ``K_POINTS automatic``.

.. note::
   Be sure to set ``nbnd`` to as many bands as you need for the Wannierization. This is not necessarily the same as the number of centers in the Wannier calculation -- we mean you should set it to the same number of bands as in the ``*.win`` file.



Step 3: Phonons and electron-phonon coupling
--------------------------------------------

Now, we use the ``ph.x`` executable from our patched QE to run a phonon calculation, during which the electron-phonon matrix elements on a coarse mesh are computed. Again, this is similar to the process from the EPA tutorial. The input file ``ph.in`` is as follows::

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


In the input file, we set the flag ``electron_phonon = "epa"``. Even though we are not doing an EPA calculation, this flag is generally used to trigger the calculation of the electron-phonon coupling matrix elements which are used by Phoebe.

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

* In the current release, we don't support spin-polarized calculations or spin-orbit calculations. Support for this will come in a later release (as we need to implement spin-related symmetries).



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

In the working folder ``./example/Silicon-epa/qe-elph``` run the command::

    /path/to/patched-quantum-espresso/bin/q2r.x -in q2r.in > q2r.out

If the code run successfully, you should see a new file ``silicon.fc``.



Step 5: Non-self-consistent run
-------------------------------

We now start the process of Wannierizing the electronic band structure.
Before running Wannier90, we need to compute the electronic band structure on the full grid of k-points as a starting point for the Wannier calculation.
You can check that the ``bands.in`` file is essentially identical to the `scf.in` file, except that we:

* Modified the calculation parameter to ``calculation = "bands"``, which indicates to QE that we will use the charge density computed in Step 2 to recompute the wavefunctions. Don't set this parameter to ``"nscf"``, as this has a slightly different output.

* Instead of using the keyword ``K_POINTS automatic, 6 6 6 0 0 0``, we explicitly write the coordinates of all :math:`6^3` k-points. These can be generated using the helper script provided by Wannier90, ``q-e/wannier90-3.0.0/utility/kmesh.pl``, run on the command line by specifying the k-mesh used in the scf calculation. For example, ``kmesh.pl 6 6 6`` will produce the k-point list.

To run it, type::

  mpirun -np 4 /path/to/phoebe-quantum-espresso/bin/pw.x -in bands.in > bands.out


Step 6: Wannierization
----------------------

Now, we can Wannierize the band structure in three steps.

First, we run Wannier90 in preprocessing mode::

 /path/to/phoebe-quantum-espresso/bin/wannier90.x -pp si

Then, we convert data from QE to Wannier90. The input file of pw2wannier90 is pretty minimal. Here we name it `pw2wan.in`.::

 &inputpp
   outdir = './out'
   prefix = 'silicon'
   seedname = 'si'
 /

And can be run by::

  mpirun -np 4 /path/to/phoebe-quantum-espresso/bin/pw2wannier90.x -in pw2wan.in > pw2wan.out

Finally, run the actual wannierization::

  mpirun -np 4 /path/to/phoebe-quantum-espresso/bin/wannier90.x si

For your future research project, make sure that ``prefix`` and ``outdir`` are consistent with the ``pw.x`` calculation above, and that ``seedname``, the string following wannier90.x, is consistent with the name of the Wannier90 input file ``{seedname}.win``.
The input file used above to run Wannier90 is a bit more involved::

	write_tb = true
	write_u_matrices = true

  !exclude_bands = 0
	bands_plot        = true

	num_bands         = 12
	num_wann          = 8
	dis_win_max       = 17.d0
	dis_froz_max      = 10.4d0
	dis_num_iter      = 120
	dis_mix_ratio     = 1.d0

	num_iter          = 500
	num_print_cycles  = 50

	begin unit_cell_cart
	bohr
	-5.1000 0.0000 5.1000
	 0.0000 5.1000 5.1000
	-5.1000 5.1000 0.0000
	end unit_cell_cart

	begin atoms_frac
	Si   0.00  0.00   0.00
	Si   0.25  0.25   0.25
	End atoms_frac

	begin projections
	Si : sp3
	end projections

	begin kpoint_path
	L 0.50000  0.50000 0.5000 G 0.00000  0.00000 0.0000
	G 0.00000  0.00000 0.0000 X 0.50000  0.00000 0.5000
	X 0.50000 -0.50000 0.0000 K 0.37500 -0.37500 0.0000
	K 0.37500 -0.37500 0.0000 G 0.00000  0.00000 0.0000
	end kpoint_path

	mp_grid = 6 6 6

  begin kpoints
    0.00000000  0.00000000  0.00000000
    ...
    0.83333333  0.83333333  0.83333333
  end kpoints

The k-point list at the end of the calculation is the same list used in the nscf calculation above, however now we do not want the kpoint weights generated by `kmesh.pl`, so instead, you should generate the k-point list using the command `kmesh.pl 6 6 6 -`. If you want to check that the Wannierization went well, you can use the output coming from the command ``bands_plot = true`` using gnuplot::

	gnuplot ./si_band.gnu --persist


.. note::
   It's important that you set the variables::

     write_tb = true
     write_u_matrices = true

   These will write to file the Hamiltonian in the Wannier representation and the rotation matrices :math:`U` that are needed to run Phoebe.

.. note::
   If you have core states which are below your disentanglement window, you MUST set exclude bands to remove these from the output Wannierization. If you don't set this, Wannier90 does not provide any information about the number of electrons used in the calculation, and you may see errors from Phoebe about your Hamiltonian not having enough bands, or find that your Fermi level in Phoebe is nonsensical.

   If you forgot to set this parameter, you can probably rectify it during the Phoebe calculation by setting the :ref:`numOccupiedStates` input variable to the number of electrons in your DFT calculation - the bands under the disentanglement window. Then make sure the Fermi level calculated by Phoebe approximately matches the one from your scf output."


The variable ``num_bands`` should match the value of ``nbnd`` set in ``scf.in`` and ``nscf.in``.

The variable ``num_wann`` is the number of Wannier functions that are used in the calculation. You should aim to Wannierize bands up to and slightly above the chemical potential. Additionally, as many loops in a transport calculation run over nBands, you also don't want to Wannierize an unnecessary number of bands.

In this input file, we provide the Wannierization disentanglement parameters and the orbital projections (the orbitals which are used as a starting guess for the Wannier orbitals).
The meaning of these quantities is described in the `Wannier90 documentation <http://www.wannier.org/support>`__.
This is the notoriously hard part of a Wannierization procedure, and every different material may present a new challenge.
The Wannier90 tutorials and manual can help you choose these parameters for your research project. For any material you Wannierize, it's very important to check the quality of the Wannierization, as with gnuplot above. Ideally, you should plot the QE band structure and the output Wannier bandstructure on top of each other to see if your calculation was successful.



Step 7: QE to Phoebe conversion
-------------------------------

Now that we have generated all the necessary outputs of QE and Wannier90, we can get started with Phoebe.
In this section, we read all the information scattered throughout the files created above and prepare the electron-phonon coupling for the transport calculation.
In this step, we transform the electron-phonon coupling matrix elements from the Bloch to the Wannier representation.

To understand how this works, let's look at the input file ``qeToPhoebeWannier.in``::

  appName = "elPhQeToPhoebe"
  elPhInterpolation = "wannier"
  phFC2FileName = "silicon.fc"
  electronH0Name = "si_tb.dat"
  wannier90Prefix = "si"
  quantumEspressoPrefix = "silicon"

The key parameters used in this calculation are:

* :ref:`appName` = `"elPhQeToPhoebe"`:
  here we select the app to post-process the electron-phonon coupling files created by the modified version of QE.

* :ref:`elPhInterpolation` = `"wannier"`:
  this selects the post-processing method used to transform the electron-phonon matrix elements. In this case, we select the method which transforms them to the Wannier representation.

* :ref:`phFC2FileName` = `"silicon.fc"`: points to the location of the harmonic force constants file created by ``ph.x``.

* :ref:`electronH0Name` = `"si_tb.dat"`: this parameter, in the form of `{wannier90seedname}_tb.dat`` should point to the file created by Wannier90 due to the ``write_tb = true`` flag. If Wannier90 has disentangled bands, there should also be a file called ``si_tb_dis.dat`` in this directory.

* :ref:`wannier90Prefix` = `"si"`: should match the ``seedname`` value of Wannier90, and it is used to locate various ``./si.*`` files.

* :ref:`quantumEspressoPrefix` = `"silicon"`: this parameter is used to locate and read the files ``./silicon.phoebe.*.dat`` that have been created by ``ph.x``. It should match the prefix parameter specified in the ``pw.x`` and ``ph.x`` input files.


There's no other parameters to tune in this part of the code -- just make sure that Phoebe can locate all these files.
To execute the code::

  export OMP_NUM_THREADS=4
  /path/to/phoebe/build/phoebe -in qeToPhoebeWannier.in -out qeToPhoebeWannier.out

and wait until completion.

Note that this calculation can be memory intensive.
For this reason, we recommend to limit/avoid use of MPI parallelization and use a large number of OMP threads (if you compiled the code with OpenMP. OpenMP is useful, because it allows multiple threads to work on a problem while sharing the memory on a node).

For some large calculations, the electron-phonon coupling tensor may be very large, so that a single MPI process cannot store an entire copy of the tensor in its own memory.
If this is the case (e.g. if some segmentation faults appear), you can try setting the input variable :ref:`distributedElPhCoupling` = `"true"`: this will decrease the memory requirements of the calculation in exchange for a slower calculation, and will parallelize with MPI over the irreducible q-points.

After the code completes, you should see an output file called ``silicon.phoebe.elph.dat`` or ``silicon.phoebe.elph.hdf5`` if you compiled Phoebe with HDF5 support (recommended).


.. _calculateElTransport:

Step 8: Electronic Transport from Wannier interpolation
--------------------------------------------------------

After all this work, it's time to run Phoebe and compute the transport properties.
The input file for computing electronic transport properties::

  appName = "electronWannierTransport"
  phFC2FileName = "silicon.fc"
  sumRuleFC2 = "crystal"
  electronH0Name = "si_tb.dat",
  elphFileName = "silicon.phoebe.elph.hdf5"

  kMesh = [15,15,15]
  temperatures = [300.]
  dopings = [1.e21]

  smearingMethod = "gaussian"
  smearingWidth = 0.5 eV
  windowType = "population"

  scatteringMatrixInMemory=true
  solverBTE = ["iterative","variational","relaxons"]


The notable parameters in this input file are:

* :ref:`appName` = `"electronWannierTransport"`: selects the app for computing electronic transport properties with Wannier interpolation.

* :ref:`phFC2FileName` = `"silicon.fc"`: points to the location of the harmonic force constants file created by ``ph.x``.

* :ref:`sumRuleFC2`: impose translational invariance on the force constants, so that acoustic phonon frequencies go to zero at the gamma point.

* :ref:`electronH0Name`: points to the ``si_tb.dat`` file created by Wannier90, which contains the electron Hamiltonian in the Wannier representation.

* :ref:`elphFileName`: is the path to the file containing the electron-phonon coupling, which was created in step 7 by ``elPhQeToPhoebe``. If you built with HDF5, this is an hdf5 file.

* :ref:`kMesh`: this specifies the mesh of wavevectors used to integrate the Brillouin zone.

* :ref:`temperatures`: a list of temperatures in Kelvin, for which Phoebe will compute transport results.

* :ref:`dopings`: a list of dopings in :math:`cm^{-3}` at which we will compute results. This is only meaningful for semiconductors.

* :ref:`smearingMethod` (and :ref:`smearingWidth`): sets the algorithm to approximate the Dirac-delta conserving energy. In this case, we are using the "gaussian" scheme, and the parameter :ref:`smearingWidth` should be converged together with the :ref:`kMesh`. Alternatively, one could use the "adaptiveSmearing" method, which chooses an adaptive width automatically. See the :ref:Theory section for more discussion.

* :ref:`windowType`: reduces the number of electronic states to only those close to the chemical potential. It selects for the electronic states such that :math:`\frac{\partial n}{\partial T} < \delta` and :math:`\frac{\partial n}{\partial \epsilon} < \delta`, where :math:`\delta` is set by :ref:`windowPopulationLimit`. This makes the calculation much faster, as only a few states close to the chemical potential are relevant for transport calculations. If Wigner corrections are important to you, see the below note.

* :ref:`scatteringMatrixInMemory`: sets the scattering matrix to be kept in memory. This speeds up the calculation, but makes it much more memory intensive.

* :ref:`solverBTE`: selects which solvers to use for the linearized BTE (i.e. solutions beyond the relaxation time approximation. The RTA solution is always computed and output, even if this variable is left unset).

To run the code, we can simply do::

  export OMP_NUM_THREADS=4
  /path/to/phoebe/build/phoebe -in electronWannierTransport.in -out ewt.out


.. note::
     Transport coefficients should be converged with respect to the :ref:`kMesh` parameter, as well as the :ref:`smearingWidth`, if the Gaussian smearing method is chosen.

.. note::

  If this calculation runs out of memory, reference the parallelization section at the bottom of the page for advice.

.. note::

  If you are interested in the Wigner contribution to the transport properties, you need to set a large energy window or set window to "none". As described in `Materials Today Physics 19, 100412 (2021). <10.1016/j.mtphys.2021.100412>`_, Fig 4., WTE contributions can come from states far away from the chemical potential. If we set a narrow window, these states will be discarded (as is appropriate for semiclassical BTE transport).

Output
------

There are two kinds of output: the standard output file (in the line above, it's ``ewt.out``) and the JSON files containing more extensive transport and lifetime values.

.. raw:: html

  <h4>Standard Output File</h4>

The main output file shows results as well as a report of the calculation progress.
The calculation progresses in this way:

* We start by parsing all input files.

* Then, the electronic band structure is computed, and filtered with the window, if needed. In this step, we also compute the Fermi level, chemical potentials, and doping concentrations.

* Next, Phoebe computes the scattering matrix, which is often the most time-consuming step.

* Using the scattering matrix, Phoebe solves the BTE at the relaxation time approximation level and computes the set of electronic transport coefficients (electrical conductivity, mobility, electronic thermal conductivity, and Seebeck coefficient).

* After the basic RTA solution is computed, we solve the Wigner transport equation at the relaxation time approximation level, and output the transport coefficients.

* The electronic viscosity is computed at the relaxation time approximation level.

* Finally, we start the exact solvers of the linearized BTE. After some time and multiple iterations of the scattering matrix, we compute the transport coefficients. For the "relaxons" solver, we also compute the electronic viscosity obtained by solving the linearized BTE, if symmetries are not used.

.. raw:: html

  <h4>JSON Output Files</h4>

There are several JSON files containing all the output, such as the electronic band structure, the electronic lifetimes/linewidths on the selected :ref:`kMesh`, and the transport properties. They also contain information which specifies that this output is for electrons, as well as the units associated which each kind of output. It's worth opening and printing the keys from each JSON file to see the information in each file.

You can learn more about how to post-process these files at :ref:`postprocessing`.

**Files which are always output for this calculation:**

* ``el_specific_heat.json``: contains the electronic specific heat.
* ``rta_wigner_coefficients.json``: contains the RTA solution with the Wigner transport correction.

**As well as a few which are output for specific solvers:**

* ``solver_onsager_coefficients.json``: contains the transport coefficients at each temperature and doping point specified in the Phoebe input file.

* ``solver_electron_viscosity.json``: contains the electronic viscosity. This can be output by the RTA solver, and for cases where Phoebe was run with ``useSymmetries = false``, for the relaxons solver as well.

* ``solver_el_relaxation_times.json``: contains the relaxation times on the :ref:`kMesh` specified in the ``electronWannierTransport`` input file. It is only output for solvers "rta" and "relaxons", as the lifetime is not well defined for the iterative solvers.

To understand how to parse these files in more detail, take a look at the scripts described by the :ref:`postprocessing` page. In particular, if you want to plot lifetimes vs. energy, look at ``tau.py``. If you want to plot the transport coefficients vs. doping or temperature, check out ``transport_coefficients.py``.

**Note:** though we output the Wigner correction with the RTA Onsager coefficients, the Wigner correction is additive for the conductivity and mobility. (Though, not additive for the Seebeck coefficient and electronic thermal conductivity). Therefore, for the electrical conductivity, for example, if you want Wigner+iterative electrical conductivity, you should take :math:`\sigma_{Wigner+RTA} - \sigma_{RTA} + \sigma_{Omini}`.

Convergence Checklist
----------------------

In this tutorial we show a demo calculation, which is certainly unconverged. We don't discuss the convergence tests that need to be done for a production/publication quality research project.

**You should make sure to test the convergence of:**

* Check that the phonon frequencies are converged with respect to k-point sampling, q-point sampling and wavefunction cutoff.

* Test the convergence of the Wannier90 bandstructure with respect to the k-point sampling. Make sure that the Wannier90 output band structure matches well the DFT band structure.

* Test that the electronic bandstructure is converged with respect to the k-point sampling, the ``ecutwfc`` (and ``ecutrho``) parameters of ``pw.x``.

* Test the convergence of the electronic transport coefficients with respect to ab-initio results, in particular with respect to the k/q-point sampling in the DFT calculation.

* Check the convergence of the electronic transport results with respect to the parameters :ref:`kMesh` and, if applicable, the :ref:`smearingWidth`.


Parallelization
----------------

The sections on parallelization discussed for the phonon transport app apply to the electronic transport app as well. In addition to this, there were a few important points mentioned in the text of this tutorial:

* **For the qeToPhoebeWannier app:** The electron-phonon coupling tensor may be very large, so that a single MPI process cannot store an entire copy of the tensor in its own memory.

  If this is the case, you can try setting the input variable :ref:`distributedElPhCoupling` = `"true"`: this will decrease the memory requirements of the calculation in exchange for a slower calculation, and will parallelize with MPI over the irreducible q-points.

* **For the electronWannierTransport app:**

  * **If the code fails while reading in the el-ph coupling HDF5 file:** If your electron-phonon tensor has become to large to fit on a single node (the case when MPI processes = 1, OMP threads = #cores/node), you can group the MPI processes into groups, (aka "pools") of processes. Each pool of processes takes a section of the tensor, parallelized over the :math:`R_e` vectors. For a more detailed description of this, see the :ref:`poolsize` section of the "Running Phoebe" section.

  * **If the code fails while allocating the scattering matrix:** :ref:`scatteringMatrixInMemory` = true speeds up calculations but requires a lot of memory. If the code fails during the memory allocation of the scattering matrix, you need to request more HPC resources. More MPI processes can help alleviate this, because the scattering matrix is distributed over MPI processes using ScaLAPACK/BLACS. Running with ``useSymmetries = true`` can also help to mitigate the issue by decreasing the size of the matrix.

* **For any calculation where memory is an issue:** To parallelize your calculation for cases where memory is an issue, set the number of MPI processes equal to the number of nodes, and set the number of OMP threads equal to the number of cores in the node. This will allow each process to use all the memory on a node, while still getting parallel performace benefit from the OMP threads. If applicable, the number of GPUs should match the number of MPI processes.

* **Optimize the MAXMEM parameter:** MAXMEM is relevant to tune the memory used during the interpolation of the el-ph coupling. For CPU-only runs, MAXMEM isn't critical to performance, and can be set to a relatively small value (e.g. 1Gb), much smaller than the memory available to each MPI process. For GPU-accelerated runs, set MAXMEM to the GPU on-board memory (e.g. ``export MAXMEM=16`` to tell Phoebe that the GPU has 16 Gb of on-board memory).
