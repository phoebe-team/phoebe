.. _shengbte:

Anharmonic Force Constants with thirdorder.py
==============================================

Overview
--------

As an alternative to phono3py, we also support anharmonic force constants generated from ShengBTE's thirdorder.py script (specifically, when used with QE as the force calculator). These constants can also be used with the phonon transport app from Phoebe.

Step 1: Run pw.x
--------------------------------------

Again, we start by calculating the total energy of the silicon unit cell. This calculation will create the ground state charge density and wavefunctions that are used in further calculations.

To run, go to the folder ``./example/Silicon-ph/qe-phonons`` in the phoebe repository.
The file ``scf.in`` is the input file for the ``pw.x`` executable, and is shown below::

  &control
    calculation = "scf"
    restart_mode = "from_scratch"
    prefix = "silicon"
    tstress = .true.
    tprnfor = .true.,
    pseudo_dir = "../../pseudoPotentials/"
    outdir = "./out"
  /
  &system
    ibrav = 2
    celldm(1) = 10.2
    nat = 2
    ntyp = 1
    ecutwfc = 30.
  /
  &electrons
    conv_thr = 1.0d-14
  /
  ATOMIC_SPECIES
    Si  28.086  Si.pz-vbc.UPF
  ATOMIC_POSITIONS alat
    Si  0.00  0.00  0.00
    Si  0.25  0.25  0.25
  K_POINTS automatic
    6 6 6 0 0 0

A detailed description of these parameters can be found on `Quantum ESPRESSO's website <https://www.quantum-espresso.org/Doc/INPUT_PW.html>`__.
The most important parameters are:

* **K_POINTS:** the parameter controlling the integration mesh of wavevectors on the Brillouin zone. Phonon properties should be converged against this mesh (more wavevectors is better). Tip: ``ph.x`` calculations are faster when the k-mesh is gamma-centered.

* **ecutwfc:** the parameter controlling the number of G-vectors used in the plane-wave expansion of the wavefunction. Phonon frequencies should be converged against this value.

* **conv_thr:** this parameter controls the total energy convergence threshold. Note that a low value of conv_thr may result in poorly converged phonon properties.

* **prefix:** prefix of some output files. Make sure to use a consistent value of prefix throughout your calculations.

* **outdir**: name of the scratch folder. Must be used consistently throughout the run so that it points to the correct files.

This list is obviously not complete, and for your research project you may need to use more functionalities from QE's ``pw.x``.

Simply run it as::

  /path/to/qe/pw.x -in scf.in > scf.out

after substituting the suitable path to the `pw.x` executable.


Step 2: Run ph.x
---------------------------------------

The input file ``ph.in`` is as follows::

  phonons of Si
  &inputph
    tr2_ph = 1.0d-14
    prefix = "silicon"
    ldisp = .true.
    nq1 = 4
    nq2 = 4
    nq3 = 4
    outdir = "./out"
    fildyn = "silicon.dyn"
  /

The values of ``nqX`` select the Monkhorst-Pack grid of q-points centered at Gamma, for which we will compute the phonon properties.
Also, it's important that ``prefix`` and ``outdir`` are the same as those used in the ``pw.x`` calculation from step 2.
Use a good value of ``tr2_ph`` (smaller is better, but harder to converge), which (indirectly) checks the convergence of phonon frequencies.

Run the code as::

    /path/to/qe/bin/ph.x -in ph.in > ph.out

If the code executes correctly and completely, you should see a number of files called ``{fildyn}*``, as many files as the number of irreducible q-points.

.. note::
   We recommend you parallelize this calculation using the ``npool`` flag from QE. This parameter controls the parallelization over k-points. It gives an almost linear speedup, but is bound by the number of k-points. For example, the below line will run QE divided into "pools" of 4 k-points::

     mpirun -np 4 /path/to/qe/bin/ph.x -npool 4 -in ph.in > ph.out


.. note::
   For a real calculation, we strongly recommend to check the convergence of the phonon frequencies with respect to the k-point mesh, the q-point mesh, the wavefunction cutoff and the ``tr2_ph`` parameter. We also recommend to using a small ``conv_thr``, as done here in the ``scf.in`` file above.


Step 3: Run q2r.x
---------------------------------------

The code ``ph.x`` has created a set of ``silicon.dyn*`` files, which contain the dynamical matrix at every irreducible q-point.
ow, we run ``q2r.x`` in order to Fourier transform the dynamical matrices in the reciprocal space representation to the real space representation, where they represent the harmonic interatomic force constants.
The input file ``q2r.in`` is minimal::

 &input
   fildyn='silicon.dyn',
   flfrc='silicon.fc'
 /

where the first variable must match the path to the dynamical matrices created earlier by ``ph.x``, and ``flfrc`` specifies the name of the output file ``q2r.x`` will create, which contains the harmonic force constants.

In the working folder ``./example/Silicon-ph/qe-phonons`` run the command::

    ./path/to/qe/bin/q2r.x -in q2r.in > q2r.out

If the code run successfully, you should see a new file ``silicon.fc``.


Step 4: Calculate Anharmonic Force Constants
---------------------------------------------

In this section, we want to use a finite-displacement approach to compute the matrix of third derivatives of the total energy with respect to ionic displacements.
To calculate these third-order force constants, we will use Quantum ESPRESSO to compute energies/forces, and a script provided by ShengBTE called ``thirdorder.py`` to generate a pattern of displacements on a supercell of the original silicon crystal. We then use QE to calculate the forces associated with these displacement patterns.

* Download ``thirdorder.py`` from `the ShengBTE website <http://www.shengbte.org/downloads>`__

* Untar the file, and cd into the ``./thirdorder`` directory that has been just created

* Modify the source code in the following way:

  * Modify line 559 of file ``thirdorder_core.c``, from ``#include "spglib/spglib.h"`` to ``#include "spglib.h"``.
  * In file ``setup.py``, set line 10 as ``INCLUDE_DIRS = ["/your/path/to/phoebe/build/spglib_src/src"]`` and line 13 as ``LIBRARY_DIRS = ["/your/path/to/phoebe/build/spglib_build"]``.

* After making these modifications, open a terminal in the ``./thirdorder`` directory and type::

    ./compile.sh

  If everything works, you should find a ``*.so`` file in the subdirectories of ``./thirdorder/build``.

* Now, go back to the Phoebe example directory ``/path/to/phoebe/example/Silicon-ph/thirdorder.py-anharmonic/``.
  Let's check the file ``supercell_template.in``::

   &control
      calculation = 'scf'
      restart_mode='from_scratch',
      prefix='silicon',
      tstress = .true.
      tprnfor = .true.,
      pseudo_dir = '../../pseudoPotentials/',
      outdir='./out',
   /
   &system
      ibrav = 0
      nat = ##NATOMS##
      ntyp = 1,
      ecutwfc = 30.
   /
   &electrons
      conv_thr =  1.0d-10
   /
   ATOMIC_SPECIES
   Si  28.086  Si.pz-vbc.UPF
   ##COORDINATES##

   ##CELL##
   K_POINTS automatic
   3 3 3 0 0 0

  As you can see, the file is the same as ``scf.in``, but with a few notable modificatons:

  * Set ``tstress`` and ``tprnfor`` to true.

  * Removed ``celldm`` (and you should remove ``alat``, if used)

  * Set ``ibrav=0``

  * Set a ``##NATOMS##`` in place of the number of atoms for the ``nat`` variable.

  * Removed ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS`` sections and replaced them with tags ``##COORDINATES##`` and ``##CELL##``.

  * Modified the k-points, as the k-point density should decrease as the inverse of the size of the supercell we will set up. In this case, we initially set a k-point mesh of 6x6x6 points, but we will set up a supercell of size 2x2x2 and thus the new supercell k-point mesh is 3x3x3.


.. note::
   If you use the ``K_POINTS gamma`` keyword, make sure you don't use the patched version of QE modified for the electron-phonon coupling, or use it with ``K_POINTS automatic``.


* Now, we generate a set of atomic displacements on a supercell. These displacements are needed to compute the third-order force constants. From the Phoebe ``example/Silicon-ph/thirdorder.py-anharmonic/`` directory, execute the following::

    ln -s /your/path/to/thirdorder_espresso.py .
    python3 thirdorder_espresso.py scf.in sow 2 2 2 -3 supercell_template.in

  In the first command, we link the script provided by ``thirdorder``. Make sure you change the path to point to the ``thirdorder_espresso.py`` script.
  Next, you can see the script takes 7 parameters:

     * ``scf.in`` - the QE input for the unit cell.

     * ``sow`` - tells the script to generate displaced supercells.

     * ``2 2 2`` - three parameters indicating the dimensions of the supercell, which here is 2x2x2.

     * ``-3`` - indicates that we only include interactions up to the third-nearest neighbor.

     * ``supercell_template.in`` - we pass the path to the supercell template discussed above.

  This script will create a lot of input files, potentially up to the cube of the number of atoms in the supercell. Therefore, choose an appropriate number of nearest neighbors (and make sure to converge the thermal conductivity against this parameter).

* It's time to run all of these supercell calculations!
  You can do this by typing in the terminal::

    for f in DISP.supercell_template.in.*; do
      mpirun -np 4 pw.x -in $f > $f.out
    done

  This step may take a while. It could be worth using ``npools`` or other parallization methods where relevant, as well as more cores if they are available to you. Note, production quality anharmonic force constant calculations can be quite computationally demanding -- this is just a demo calculation.

* Finally, we postprocess all these forces by typing::

    # find and sort the files
    find . -name 'DISP.supercell_template.in.*out' | sort -n | python3

    # collect (aka reap) the forces from each pw.x output file
    thirdorder_espresso.py scf.in reap 2 2 2 -3

  Note here that you should use the same parameters (here, 2 2 2 -3) used for generating the supercell displacements.
  You should see a new file called ``FORCE_CONSTANTS_3RD`` with the desired output.

If all went well, you will have computed the ab-initio matrix of third-order force constants.


To run a calculation with Phoebe using these force constants, we would use an input file similar to the following::

  appName = "phononTransport"

  phFC2FileName = "qe-phonons/silicon.fc"
  phFC3FileName = "thirdorder.py-anharmonic/FORCE_CONSTANTS_3RD"

  sumRuleFC2 = "simple"

  qMesh = [6,6,6]
  temperatures = [300.]

  smearingMethod = "gaussian"
  smearingWidth = 10. cmm1

  solverBTE = ["variational"]
  scatteringMatrixInMemory = true

  boundaryLength = 1. mum
  windowType = "population"

Where the key difference from the file used in the phonon transport app is that the ``phFC2FileName`` and ``phFC3FileName`` now point to the files generated by ``q2r.x`` for the harmonic forces and ``thirdorder.py`` for the anharmonic forces.

Phoebe will then produce the desired transport output in the same way as for the previous tutorial using phono3py inputs.
