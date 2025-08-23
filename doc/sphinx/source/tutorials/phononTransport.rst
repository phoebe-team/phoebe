.. _phononTransport:

Phonon Transport Tutorial
=========================

Synopsis
--------

In this tutorial, we will use Phoebe to compute the lattice thermal conductivity of silicon. Here, we use Quantum ESPRESSO for the calculation of interatomic force constants. However, in practice, `any DFT package that works with Phono3py <https://phonopy.github.io/phono3py/interfaces.html>`__ could be used.

As described in the :ref:`Theory` section of this manual, we need Quantum ESPRESSO to compute the phonon dynamical matrix and the third derivative of the total energy with respect to ionic displacements (aka the anharmonic or third-order force constants.) In this case, it shouldn't matter which version of QE you use, and it's not necessary to use a patched copy of QE.

Though we again assume the user is familiar with QE, several tutorials can be found on `Quantum ESPRESSO's website <https://www.quantum-espresso.org/resources/tutorials>`__, which cover the DFT aspects of the calculation in more detail.

Though we show below how to do this calculation with phono3py, Phoebe is also capable of taking anharmonic force constants from ShengBTE used with QE, as generated from the `thirdorder python script <https://www.shengbte.org/development>`_ associated with ShengBTE. We support the use of thirdorder.py with QE only, for the time being. See :ref:`shengbte` for details.

.. note::
  Support of phono3py is only available if Phoebe is built using `hdf5`. It's needed for this tutorial.

.. _phono3pyInstallation:

Step 1: Phono3py Installation
-----------------------------

To calculate the anharmonic force constants from phono3py, you first need to follow the instructions to `set up phono3py <https://atztogo.github.io/phono3py/install.html#installation-from-source-code>`_. To summarize the brief installation process, you first need to have Anaconda installed on your computer (which you can `download and install from here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_). Then, you can install phono3py in a conda environment with the following on the command line instructions::

  # create a conda environment named phono3py
  conda create --name phono3py python=3.10
  # activate the enviroment (use conda activate for newer conda versions)
  source activate phono3py
  # install phono3py in this environment
  conda install -c conda-forge phono3py

You should now have the command `phono3py` available to run on the command line.
If for any reason this doesn't work, refer to the phono3py documentation linked above for more detailed installation instructions.

.. _forceConstantCalculation:

Step 2: Calculation of Force Constants
---------------------------------------

The procedure to calculate the force constants with phono3py differs slightly depending on which DFT package you want to use. The procedures for interfacing with different calculators are available `here <https://phonopy.github.io/phono3py/interfaces.html#>`_.

However, the procedure to follow is similar in all cases. Below, ``<input-file-name>`` should be a ``POSCAR`` in the VASP case or a ``pw.x`` style ``pw.in`` file in the QE case. ``<DFT-package-name>`` will be ``qe`` for Quantum Espresso. The package name input flag is not necessary for VASP, which is used by default (though ``--vasp`` will work). For other codes, look at the examples in the above link for input file and package flag information.

Before beginning the calculation, we strongly recommmend you transform whatever unit cell you want to use in the calculation to the primitive cell which will be used internally by phono3py. This avoids the use of flags which signal the need to transform to this primitive cell (see flag `--pa <https://phonopy.github.io/phono3py/command-options.html#pa-primitive-axes-primitive-axes>`_). If your chosen cell wasn't primitive, this can also reduce the cost of an already expensive calculation::

  phonopy --<DFT-package-name> --symmetry -c <input-file-name>

The primitive cell will be written in the same format in as your input, in a file named ``Punitcell.in``, ``PPOSCAR``, or whichever generate file starts with "P" for "primitive". Use this structure as ``<input-file-name>`` for the rest of the tutorial. If you're using QE, the output ``Punitcell.in`` file needs to be copied back into your ``scf.in`` file, replacing the original structure parameters.

.. note::

  A full-scale anharmonic force constant calculation can be expensive! Before running a real anharmonic calculation, we recommend checking the quality of your phonon dispersion using phonopy. See below for :ref:`harmonic_p3py`.

Now, we generate the displaced supercells (of the specified dimensions, here 2x2x2) using the command::

  phono3py --<DFT-package-name> -d --dim="2 2 2" -c <input-file-name>

You should see that this command generated a large number of files.
The first file is ``supercell.in`` which contains the positions of a super cell with atoms on undisplaced positions.
There should also be a large number of files named ``supercell-$i.in`` where ``$i`` runs from 1 to N (this number `N` depends on the details of your calculation). The situation will be similar for other DFT packages, though the generated structure files will match those of the chosen package.

For each of these super cells with displaced atoms, run the DFT calculation for each of these configurations.


.. raw:: html

  <h3>Note for QE users</h3>


These files generated by phono3py only contain the geometric parameters of the supercell, which must be copied to QE input files. Therefore:

  * Copy the QE input file containing the unit cell (``<input-file-name>``) to a new file, ``template.in``.

  * In ``template.in``, remove the blocks describing ``ATOMIC_SPECIES``, ``CELL_PARAMETERS``, and ``ATOMIC_POSITIONS``. Also, correct the ``nat`` flag with the value you see in the first line of file ``supercell.in``. Don't forget to set ``tprnfor=.true.`` in order to compute forces, and to set the k-points for the supercell to be smaller than those used for the primitive unit cell.

  * To generate the input files for each of these configurations you can use the below script::

      cp template.in disp.in
      cat supercell.in >> disp.in
      for i in $(seq -f "%05g" 1 N); do
        cp template.in disp-$i.in
        cat supercell-$i.in >> disp-$i.in
      done

    where you should replace `N` with the number of ``disp-`` files created by phono3py.

You should now have a number of ``disp-*.in`` files, with the QE input files that must be launched, which you can run using the loop::

  mpirun -np 1 pw.x -in disp.in > disp.out
  for i in $(seq -f "%05g" 1 N) ; do
    mpirun -np 1 pw.x -in disp-$i.in > disp-$i.out
  done

Take care to modify these lines to parallelize QE as best you can. If you want to run each displacement calculation as an independent job, as would be sensible for a full scale calculation, you must modify this script to copy the displacement files into unique directories, and run from within them. Otherwise, simultaneous jobs might overwrite one another.

After all these calculations have run, you will have all the files needed for the next step.


Step 3: Construct Force Constant Matrices
------------------------------------------

Once all calculations are finished, collect the force constants from them using a line like the following, where ``disp-{00001..nCalculations}.out`` is a list of all the output files generated in Step 2::

  phono3py --<DFT-package-name> --cf3 disp-{00001..nCalculations}.out

This creates a file named ``FORCES_FC3``, which contains the force constants. To use this information as an input to Phoebe, run the following line to compress this information into two DFT-package independent hdf5 files, ``fc2.hdf5`` and ``fc3.hdf5``, which contain the second and third order force constants, respectively::

  phono3py --<DFT-package-name> --dim="2 2 2" -c <input-file-name> --sym-fc

Before proceeding, you should check the quality of the calculation. First, make sure the harmonic phonon bands look appropriate by saving the below input to an input file (here, we'll call it ``phononBands.in``::

  appName = "phononBands"

  # necessary input files
  phFC2FileName = "fc2.hdf5"
  phonopyDispFileName = "phono3py_disp.yaml"

  sumRuleFC2 = "simple"
  begin point path
  ...
  end point path

To include the non-analytical correction to the FC2s, be sure to supply the :ref:`phonopyBORNFileName` parameter as well.

Then run a very inexpensive band calculation with Phoebe::

  mpirun -np 1 /path/to/phoebe -in phononBands.in

and plotting the resulting phonon dispersion with the ``bands.py`` script found in Phoebe's ``scripts/plotScripts/`` directory, as described in the :ref:`postprocessing` section. If everything looks as expected, continue on with the calculation.

.. note::
  You should make sure this disperson is converged with respect to DFT parameters (energy cutoff, kpoint mesh, etc) and also with respect to the dimension of the supercell provided to phono3py. It is also recommend you check the convergence of the final calculated transport properties with respect to supercell size.

.. _calculatePhononTransport:

.. _thermalConductivityCalculation:

Step 4: Calculate Lattice Thermal Conductivity
------------------------------------------------

If this dispersion looks good, we are now ready to move on to phonon transport calculations using Phoebe.

There are three files output by phono3py which we will need: ``fc2.hdf5``, ``fc3.hdf5``, ``phono3py_disp.yaml``.
The first two files contain the harmonic and anharmonic force constants, while the ``phono3py_disp.yaml`` describes the crystal supercells.

Any of the phonon related apps can be run with these files, including the phononBands, phononDos, and lifetime apps. We describe here the use of the transport app here, but the input for other apps will be similar.

Now, we are ready to use Phoebe to calculate the lattice thermal conductivity. The Phoebe input file will look something like::

  appName = "phononTransport"

  # below lines specify the paths to phono3py input files
  phFC2FileName = "fc2.hdf5"
  phFC3FileName = "fc3.hdf5"
  phonopyDispFileName = "phono3py_disp.yaml"
  useSymmetries = true

  sumRuleFC2 = "crystal"
  qMesh = [10,10,10]
  temperatures = [300.]
  smearingMethod = "adaptiveGaussian"
  solverBTE = ["iterative"]


Let's go through these parameters:

* :ref:`appName` = `"phononTransport"` triggers the calculation of phonon transport properties.

* :ref:`phFC2FileName` must point to the harmonic forces constants file.

* :ref:`phFC3FileName` must point to the third-order force constant file.

* :ref:`phonopyBORNFileName` optionally, can point to the file in the phonopy BORN format to supply parameters from non-analytic correction.

* :ref:`phonopyDispFileName` must point to the ``phono3py_disp.yaml`` file and is needed to read the force constant files.

* :ref:`useSymmetries` turns on the use of symmetries, which reduces the computational cost. If you want to run relaxon or variational solvers, you need to turn this off instead.

* :ref:`sumRuleFC2` allows us to re-enforce the translational-invariance of force constants, which is broken by numerical inaccuracy. After imposing this sum rule, acoustic phonon frequencies should go to zero at the gamma point.

* :ref:`qMesh` specifies the size of the grid of wavevectors used to integrate the Brillouin zone. Note that the value used here is very unconverged, so that the example can finish in a short amount of time.

  .. note::
     Results must be converged against values of :ref:`qMesh`!

* :ref:`temperatures` sets a list of temperatures for the calculation, in Kelvin.

* :ref:`smearingMethod` sets the algorithm to approximate the Dirac-delta conserving energy. Using the "adaptiveGaussian" scheme is particular convenient as the width of the Gaussian used to represent delta functions is automatically adjusted. The fixed-width "gaussian" scheme is also available -- in this case, you must set the :ref:`smearingWidth` parameter (and converge w.r.t. it).

* :ref:`solverBTE` selects the algorithm to solve the linearized Boltzmann transport equation. If no algorithm is specified, we only compute results within the relaxation time approximation. Above, we are only using the default RTA calculation and the additional iterative solver to find the solution to the BTE.

With this input, we can compute the phonon contribution to thermal conductivity of silicon. We run this calculation using Phoebe::

  export OMP_NUM_THREADS=4
  mpirun -np 1 /path/to/phoebe/build/phoebe -in phononTransport.in > phTransport.out

.. note::
   By default, isotopic scattering at natural abundances is included in the scattering matrix. To disable or modify it, check the parameters :ref:`withIsotopeScattering` and :ref:`masses`.

.. note::
   In several studies you may want to include boundary scattering. To include it, use the parameter :ref:`boundaryLength`.


Output
------

As usual, there are two kinds of output: the standard output file (in the line above, it's ``phTransport.out``) and the JSON files containing more extensive transport and lifetime values.

.. raw:: html

  <h4>Standard Output File</h4>


This file shows results as well as a report of the calculation progress. The structure of the calculation follows as:

* Parsing input files.

* Computing the phonon band structure (energies, eigenvectors and velocities).

* Computing the scattering matrix (this takes place whenever you see a block like this one)::

    Started computing scattering matrix with 64 q-points.
    2020-10-30, 09:15:02 |   1% |  1 / 64
    2020-10-30, 09:15:02 |   4% |  3 / 64
    2020-10-30, 09:15:02 |   9% |  6 / 64 | remaining: 6.62e-01 s.
    ......
    2020-10-31, 09:15:03 | 100% | 64 / 64 | remaining: 2.50e-02 s.
    Elapsed time: 0.81 s.

  where, for your convenience, we try to estimate the time to completion.

* Calculation of the thermal conductivity within the relaxation time approximation.

* Calculation of Wigner thermal conductivity, obtained including off-diagonal contributions of the flux operator calculated from the RTA level linewidths. 

* Calculation of the thermal viscosity tensor within the relaxation time approximation.

* Calculation of the lattice contribution to specific heat (at constant volume).

* Optional: if you selected an exact solver, you will see additional output including iterative solutions to the BTE, which includes the thermal conductivity obtained by solving the full linearized BTE (including off-diagonal matrix elements of the scattering operator).

* Optional: if you use the relaxon solver, you will see output related to the diagonalization of the scattering matrix to calculate the thermal conductivity. (If ``useSymmetries = false``, you will also see the thermal viscosity obtained by solving the BTE exactly).

.. raw:: html

  <h4>JSON Output Files</h4>

There are several JSON files containing all the output, such as the phonon band structure, the phonon lifetimes/linewidths on the selected :ref:`qMesh`, and the transport properties. They also contain information which specifies that this output is for phonons, as well as the units associated which each kind of output. It's worth opening and printing the keys from each JSON file to see the information in each file.

You can learn more about how to post-process these files at :ref:`postprocessing`.

**Files which are always output for this calculation:**

* ``specific_heat.json``: contains the phonon specific heat.

* ``rta_phonon_viscosity.json``: contains the phonon viscosity at RTA level.

* ``rta_phonon_thermal_cond.json``: contains the phonon thermal conductivity at the RTA level.

* ``rta_ph_relaxation_times.json``: contains the RTA phonon lifetimes on the :ref:`qMesh` specified in the input file.

* ``rta_wigner_coefficients.json``: contains the RTA transport coefficients + the Wigner transport correction.

**As well as a few which are output for specific solvers:**

* ``solver_phonon_viscosity.json``: contains the electronic viscosity. This can be output by the RTA solver, and for cases where Phoebe was run with ``useSymmetries = false``, for the relaxons solver as well.

* ``solver_phonon_thermal_cond.json``: contains the phonon thermal conductivity output by a specific solver.

* ``solver_relaxation_times.json``: contains the phonon relaxation times on the :ref:`qMesh` specified in the ``phononTransport`` input file. It is only output for solvers "rta" and "relaxons", as the lifetime is not well defined for the iterative solvers.


To understand how to parse these files in more detail, take a look at the scripts described by the :ref:`postprocessing` page. In particular, if you want to plot lifetimes vs. energy, look at ``tau.py``. If you want to plot the thermal conductivity vs. temperature, check out ``transport_coefficients.py``.


Convergence Checklist
----------------------

In this tutorial we show a demo calculation, which is unconverged for the sake of a quick example. We summarize the parameters which the outputs should be converged against below.

**You should make sure to test the convergence of:**

* Test that the electronic bandstructure is converged with respect to the k-point sampling, the ``ecutwfc`` (and ``ecutrho``) parameters of ``pw.x`` before proceeding to the anharmonic force constant calculation.

* Check that the phonon frequencies are converged with respect to k-point sampling, q-point sampling, and wavefunction cutoff.

* Test the convergence of the phonon transport coefficients with respect to the size of the phonon supercell used in the anharmonic force constant calculation.

* Check the convergence of the phonon transport results with respect to the parameters :ref:`qMesh` and, if using the fixed-width Gaussian smearing method, the :ref:`smearingWidth` parameter.



Parallelization and performance
-------------------------------

As mentioned above, for the ``qeToPhoebe`` calculation, the primary method of parallelization is over OMP threads, as this calculation can be memory intensive, and OMP helps to alleviate this. For this reason, we've written the code to be sped up when using more OMP threads.

For the transport Phoebe calculation, the bottleneck is typically the construction of the scattering matrix (or the evaluation of a scattering matrix-vector product). If you are not familiar with parallelization techniques, you should read up on `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`__ and `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__.

Phoebe takes advantage of three different parallelization schemes for the phonon transport calculation.

* **MPI parallelization.** We distinguish two cases. If we want to compute the action of matrix :math:`\sum_{k'b'} A_{k,k',b,b'} f_{k'b'}`, we MPI-distribute over rows of wavevectors to achieve the best performance. If we want to store the matrix in memory, we parallelize over pairs of wavevectors using the ScaLAPACK layout. This distributes the scattering matrix in memory, reducing the required memory per process, and also speeds up operations on the matrix.

* **Kokkos acceleration.** The calculation of the phonon-phonon coupling required by the phonon transport app can also be accelerated with Kokkos. Depending on your architecture and installation parameters, Kokkos will either run on GPUs, or CPUs with OpenMP acceleration.
  Especially for GPU-accelerated runs, remember to optimize the environment variable ``export MAXMEM=4`` in the job submission script, or in the command line, to set the available GPU on-board memory (4GB in this example). For CPU-only runs, ``MAXMEM`` instead can be set to a small value of memory, smaller and not greater than the memory available to a MPI process.

* **OpenMP parallelization.** The summations over band indices when computing the scattering rates is accelerated using OpenMP. This can be accelerated by increasing the environment variable ``OMP_NUM_THREADS``.

**A basic setup using these parameters:**

* Set the number of MPI processes equal to the number of computing nodes you are requesting. This will give each MPI process access to the memory of an entire node, which can be useful for real calculations where memory often becomes an issue.

* Set the number of OpenMP threads equal to the number of physical cores available on each computing node. This will accelerate the band summations while still having these processes share the memory of the node.

* Compile Phoebe with Kokkos. If you do so, make sure that the number of GPUs you are using matches the number of MPI processes. If you don't have a GPU, Kokkos can still accelerate the phonon-phonon calculations via the number of OpenMP threads you've set.
  For CPU-only runs, set MAXMEM to a value smaller or at most equal to the total memory available to a MPI process (in this case MAXMEM has a small impact on performance, but increases significantly memory usage).
  For GPU-accelerated runs, set MAXMEM equal to the total memory available on the GPU.


Tradeoff between speed and memory
----------------------------------

There's a parameter :ref:`scatteringMatrixInMemory` that you need to consider.
If we set this parameter to true, we store the scattering matrix in memory.
If false, we only compute the action of the scattering matrix, without ever storing all of it in memory.

There is no `best` choice here, rather, you should decide what's best for your case and decide which tradeoff works best for you.

* **Option 1:** :ref:`scatteringMatrixInMemory` = true. The scattering matrix occupies :math:`16 (3 N_{atoms} N_{q-points})^2 / 1024^3` Gigabytes, if no window is used. This number can be pretty large (even Terabytes), and you should make sure that your HPC allocation has enough memory for storing this large matrix. Given the size, we only allow you to run the code with a single temperature.

  In exchange, iterative or variational solvers of the BTE are extremely cheap, and the cost of your simulation is largely the cost of constructing the scattering matrix. Moreover, this allows you to run :ref:`solverBTE` = "relaxons" type of BTE solver.

* **Option 2:** :ref:`scatteringMatrixInMemory` = false. The memory footprint is much lighter (the square root of Option 1), so that the same calculation can be run on fewer CPUs. You can compute the thermal conductivity for multiple temperatures in the same run. The calculation of properties within the relaxation time approximation is as expensive as above (if you're only trying to calculate RTA properties, definitely select this option and save on memory).

  In exchange, iterative or variational BTE solvers are much slower. In fact, at each iteration you need to recompute the scattering matrix.
The cost of the calculation therefore grows linearly with the number of iterations of the iterative solver (which may be significant).
You also cannot diagonalize the scattering matrix as required by :ref:`solverBTE` = "relaxons", so this solver is only available with option 1.


Low temperature thermal conductivity
------------------------------------

At low temperatures, only phonons with small energies are thermally excited and most states are empty.
However, if we use use an input file like the one discussed above, we sample and sum over all phonon states, even the empty ones.
As a result, we end up spending a lot of time computing phonon states that don't contribute to transport.

To avoid doing this unnecessary work, we have the parameters :ref:`windowType`, :ref:`windowEnergyLimit`, and :ref:`windowPopulationLimit`.
If we wanted to add these two parameters to the input file above::

  windowType = "population"
  windowPopulationLimit = 1.0e-6
  temperatures = [3.]
  qMesh = [40,40,40]

Here, we are discarding all phonon states whose equilibrium occupation number is smaller than 1.0e-6.
At low temperatures, this will discard phonon modes away from the Gamma point or optical modes that are too high in energy to be thermally excited, and could mean a significant reduction in computational expense.
As a result, we can increase the values of :ref:`qMesh`, so that we can accurately sample the points close to the Gamma point. This reduction in cost is especially important, as it can require a much finer q-mesh to converge low temperature conductivities.


.. _harmonic_p3py:

Calculation of Harmonic-Only Phonopy Dispersion
-------------------------------------------------

It can be helpful to check the quality of the harmonic phonons before running the full anharmonic calculation. If you want to use phonopy's native script to check the bands without running the full anharmonic calculation, you can run::

  phonopy --<DFT-package-name> -d --dim="2 2 2" -c <input-file-name>

Then, after running scf calculations for each of the structure files generated by the above line, collect the forces::

  phonopy --<DFT-package-name> -f disp-{001..nCalculations}.out

To plot the dispersion, we'll need a ``band.conf`` file, which should contain at a minimum the high symmetry band path in crystal coordinates (with other optional settings `here <https://phonopy.github.io/phonopy/setting-tags.html#band-structure-related-tags>`_). For silicon, a simple example would be::

  # save as band.conf
  ATOM_NAME = Si
  DIM = 2 2 2
  # list bandpath high sym points here
  BAND = 0.0 0.0 0.0   0.0 0.5 0.5    0.25 0.75 0.5    0.5 0.5 0.5  0.0 0.0 0.0  0.375 0.750 0.375

Then, run the following line and check the output plot, named ``band.pdf``::

  phonopy --<DFT-package-name> -p -s band.conf -c <input-file-name>


.. _different_supercells_ph:

Running Phoebe with different fc2/fc3 supercells
------------------------------------------------

Sometimes, one needs a larger supercell to converge the harmonic force constants than the anharmonic force constants.
In this case, it is possible to run phono3py with different supercell sizes for the harmonic and anharmonic phonon force constant calculations, as shown in this `phono3py example <https://phonopy.github.io/phono3py/vasp.html>`__.

To run phonon transport in Phoebe in this case, it is simply necessary to link the file ``phono3py_disp.yaml`` generated by phono3py in the phoebe varible ``phonopyDispFileName``.
That file contains the information on the supercell size of the harmonic and anharmonic force constants.

Keep in mind however, while it is perfectly acceptable for harmonic and anharmonic force constants to be computed from different supercell sizes, the underlying primitive unit cell must still be the same (e.g. one should not use a supercell of the primitive unit cell for the harmonic force constants and a supercell of the conventional unit cell for the anharmonic force constants).
If Phoebe is run with these non-matching unit cells force constants, the strength of the phonon-phonon scattering will not be correct, and will lead to wrong results.



..
   If you run the phono3py calculation with different sized supercells, you will generate a slightly different set of files:

     * ``disp_fc2.yaml``
     * ``disp_fc3.yaml``
     * ``phono3py_disp.yaml``
     * ``fc2.hdf5``
     * ``fc3.hdf5``

   Where we can see ``disp_fc2.yaml`` is new. To run a Phoebe transport calculation with different sized fc2/fc3 unit cells, one simply needs to slightly alter the way the input files are supplied to the calculation. Simply subsitute the following lines for those typically used to specify file locations in your Phoebe input file::

     # displacement information files
     # note, dispFC2FileName is a new variable
     phonopyDispFileName = "phono3py_disp.yaml"
     dispFCFileName = "disp_fc3.yaml"
     dispFC2FileName = "disp_fc2.yaml"
     # force constant files, as usual
     phFC2FileName = "fc2.hdf5"
     phFC3FileName = "fc3.hdf5"



.. _mixing_phonopy_shengbte:

Using Phonopy for fc2 and ShengBTE for fc3
------------------------------------------

In some circumstances, one may want to use phonopy to compute the harmonic force constants of a material, and then use the ``thirdorder.py`` script provided by ShengBTE for the calculation of anharmonic force constants.

To launch a phonon transport calculation in this case, you only need to change the input parameters describing the input files.
For example::

     # this points to phonopy_disp.yaml, not phono3py_disp.yaml
     phonopyDispFileName = "phonopy_disp.yaml"
     # force constant files, from phonopy and thirdorder.py
     phFC2FileName = "force_constants.hdf5"
     phFC3FileName = "FORCE_CONSTANTS_3RD"

Here, ``phFC2FileName`` is pointing to the file path created by Phonopy that contains the harmonic force constants, and phonopyDispFileName to its associated file containing the information on the crystal.
Note that we only support reading of HDF5 files from Phonopy.
The ``phFC3FileName`` instead points to the force constants file created by ``thirdorder.py``.
Make sure that both fc2 and fc3 are computed using the same unit cell, or results will be unpredictable.
