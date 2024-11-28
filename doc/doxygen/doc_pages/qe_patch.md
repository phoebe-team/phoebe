@page QEPATCHDOCS Quantum ESPRESSO patch

As explained in the theory section about the electron-phonon coupling with Wannier interpolation, we modified QE to fix a gauge on the wavefunction.

@section fcode Fortran code

The code is modified on a separate git repository available [at this link](https://github.com/phoebe-team/phoebe-quantum-espresso/).
This repository is a fork from the official QE repository.
To develop a new patch or update it to the latest QE version, remember to pull from the remote quantum espresso repository.
For the first Phoebe release, we only patched the latest QE 6.6 version, although a retrofit should be doable.

**Files changed**

For the `pw.x` part of the code, we modified the file `PW/src/c_bands.f90`.
There are several subroutines in this file.

* First, skim through `c_bands()`. This subroutine has a loop over k-points. At each k-point, the Bloch Hamiltonian is built and diagonalized.

* The diagonalization happens in `diag_bands()`. Right after the diagonalization has been done, we call our custom subroutine `set_wavefunction_gauge()`. So this function is called at every k-point separately.

We do several things in our subroutine.

1. First, we rotate the wavefunction such that the G=0 plane wave coefficient is real and positive (with some care for degenerate bands). 

2. The first time this function is called, we analyze k-points and their symmetries.

3. Lastly, we distinguish two cases: if we are doing a `scf` calculation, we write the wavefunctions to file, imposing that these are only computed on the irreducible kpoints. If not `scf`, we read the irreducible wavefunction from file, compare it with the current wavefunction, and rotate the current wavefunction to carry the same phase of the irreducible one and to obey the symmetry operations, as described in the theory section.

As for the phonon code, we modify it in a few parts.
Two of these are easy:

1. `PHonon/PH/phq_readin.f90` here we simply allow for the calculation of electron-phonon coefficients also for semiconductors if we trigger the `epa` input variable.

2. `PHonon/PH/run_nscf.f90` this file sets up the nscf calculations needed by PH. Here, we modify an input variable of `setup_nscf` which disables symmetries on the k-points: while the calculation uses irreducible q-points, now it will use all points without symmetries in the k-point grid.

3. `elphon.f90` is a bigger modification, but not too difficult. The first time our subroutine `elphfil_phoebe` is called we analyze the symmetries of the q-point grid. This subroutine is called once for every q-point calculation. Next, we unfold the symmetries of the g coupling. Note that we also need to put it in a cartesian basis. Then, we write to file the quantity \f$ g(k,q^*) \f$, where k runs on the full grid of k-points, and \f$ q^* \f$ is the star of points that are symmetry equivalent to the current irreducible q point that is being computed. These output files are those that are passed as input to phoebe.




@section CREATEPATCH Create+apply the patch

**Code modifications**

First, download the repository phoebe-quantum-espresso from the MIR github page.
Most likely, you need to pull from the official quantum-espresso repository into our fork.
To this aim, remember to also pull the tags from the official QE repository doing this:
~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
cd path/to/fork
git checkout master
git remote add upstream https://github.com/QEF/q-e
git fetch --tags upstream
git push -f --tags origin master
~~~~~~~~~~~~~~~~~~~~~~~~~~~



Checkout the branch with the version of QE that we want to patch, for example, v6.6 is stored in the branch `qe-6.6`.
Create a new branch from `qe-6.6`, which we call `patched-qe-6.6`, and modify the source code with the lines of code needed by phoebe.
Alternatively, you may create the new branch `patched-qe-6.6` from an older patched branch, e.g. `patched-qe-6.5`, and pull the new commits of `qe-6.6` into `patched-qe-6.6`.

**Patch creation**

Assume now that the source code in branch `patched-qe-6.6` is ready to be distributed.
To create the patch, we simply save the `git diff` between our branch and the original branch released by the QE community.
Simply type
~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
git diff qe-6.6..patched-qe-6.6 > patchfile.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This will create the file `"patchfile.txt"`, containing the patch.
You may inspect this file to make sure it didn't add unnecessary modifications.
Currently only 4 files are modified by the patch.


**Patch application**

To apply the patch, go to a folder with a copy of quantum espresso to be patched. In that folder, copy `"patchfile.txt"` and execute the command

~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
patch -p1 < patchfile.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command executes successfully if the output looks like:

~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
patching file PHonon/PH/elphon.f90
patching file PHonon/PH/phq_readin.f90
Hunk #1 succeeded at 911 (offset 4 lines).
patching file PHonon/PH/run_nscf.f90
patching file PW/src/c_bands.f90
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the patch is successfully applied, compile QE and use as described in the tutorials.
The patch is tested to work in QE v6.5 and v6.6.
The first patch has been developed with Quantum ESPRESSO v6.6.
The patch thus may or may not work with older versions, we don't provide any support with that.

Note also, the user could also directly download the source code of branch `patched-qe-6.6` and use that.



**Troubleshooting**

The patch fails if the output looks like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
patching file PHonon/PH/elphon.f90
Hunk #1 succeeded at 1444 (offset -85 lines).
Hunk #2 succeeded at 1458 (offset -85 lines).
patching file PHonon/PH/phq_readin.f90
Hunk #1 FAILED at 907.
1 out of 1 hunk FAILED -- saving rejects to file PHonon/PH/phq_readin.f90.rej
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, one should manually open the `"*.f90.rej"` file and manually apply the modification in the file `"*.f90"`.
Note that the patch consists in added code. In the file c_bands.f90, we simply add a new subroutine and its call. In the phonon code, we modify the behavior of the 'epa' calculation so that data are written in phoebe format.
Generally, we only added new lines to the code or added new variables, so, as a guideline to fix a failed patch, add the extra lines/variables that appear in the diff. Do so for every file where the patch has failed.

