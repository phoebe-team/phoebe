FAQ and Performance Tips
================================================

Frequently asked questions 
----------------------------

**Why does my iterative or variational BTE solution not converge, or my relaxons solution have negative eigenvalues?** 

If the iterative solver, or especially the variational solver, does not converge, this could be a sign that your scattering matrix is poorly conditioned. You can check this by running the relaxons solver and noting if there are negative eigenvalues (which indicate the matrix is not positive semi-definite as it should be). This could indicate a number of problems:

	* For electron and phonon calcultions, you may have chosen a smearing value which violates the conservation of energy too strongly
	* For an el-ph calculation, your Wannierization may not be very good, producing noisy el-ph matrix elements, energies, etc. For phonon only calculations, you might have poor force constants. 
	* Symmetrizing the scattering matrix might help - but if you have many negative eigenvalues, likely this is just going to partially cover up a potentially serious problem. 


**Can I use the rotationally invariant ASR?**

	* One can take advantage of an improved ASR using the matdyn tool of QE. We recommend this for 2D materials. Here, one should use the ASR=‘all’ tag in matdyn. This should output the force constants using rotational invariance + vanishing stress conditions. Then, when Phoebe is run, use sumRuleFC2 = "no", as applying a further ASR will just add additional noise. 
	https://www.quantum-espresso.org/Doc/INPUT_MATDYN.html#idm17


** I have encountered the error `Error in routine phoebe (1): atom mapping failed`` when running the `phoebe-quantum-espresso` version of `ph.x`. 

This is a somewhat rare issue that emerges either because of a general issue with the crystal structure or because of a difference in how symmetries are used in the pw.x and ph.x calculations (often, it seems to appear when fractional translations are found in the sym ops list).
Typically this can be resolved by a standardized representation of your crystal structure -- generated for example by running `AFLOW <http://www.aflowlib.org/aflow-online/>`_ on your structure, using the button for "structure conversion" and "standard primitive" (or any other equivalent utility.)


Tips for running Phoebe efficiently
-----------------------------------

Phoebe is written to allow a number of options for improved computational performance. Below we list some ways you can reduce the cost of a calculation. 

**Efficient choices in the DFT calculation:** 

	* Try not to Wannierize more bands than necessary. While it is very important to have a high quality Wannierization with small spreads, Wannier interpolation involves matrix products with the unitary rotation matrices from the Wannier calculation. The number of bands in the Wannier interpolation can therefore make a big impact on the calculation cost. 

**Efficient memory use for electron-phonon calculations:** 

	One should review the notes about poolsize here: 
	https://phoebe.readthedocs.io/en/develop/running.html#electron-phonon-parallelization

	This allows the use to split up the electron phonon matrix elements over MPI processes, distributing them in memory rather than each process keeping a copy in memory. This will result in a slight slowdown due to greater proccess communication during some parts of the code -- therefore, you should try to only use as many pools as you need, and not more. 

**Efficient use of OpenMP and MPI:** 

	* Phoebe contains extensive OMP parallelism, and in almost all cases, running your calculation as a hybrid MPI-OpenMP calculation will be beneficial. 
	* For most systems, Kokkos will throw a warning telling you to set the following parameters, which can sometimes improve performance:: 

		export OMP_PROC_BIND=spread
		export OMP_PLACES=threads

	* On some systems, when using mpirun with Phoebe it's also important to include "--bind-to " in order for OMP parallelism to be used effectively, such as `mpirun -np 2 --bind-to none ...`. However, this can vary from system to system. 
	* If using `srun`, slurm typically sets the binding options appropropriately and you will see the expected OMP speedup. 
	* Regardless, you should do a few quick tests to ensure you're getting performance boosts as expected on your cluster, as OMP can make a big difference. 

**For efficient transport calculations in general, using the following is beneficial**::

	windowType = "population"
	useSymmetries = true
	windowPopulationLimit = 1e-10

* While the :ref:`windowPopulationLimit` variable is by default set to 1e-10, which should be a very safe value, in principle you may find you can reduce calculation cost by increasing this value -- however, you should be careful to test convergence against this parameter if you choose to do so. 

* However, there are two important caveats: 
	* If you want to use the relaxons or variational solvers, you cannot use symmetries and this must be set to false. 
	* If you are trying to use the electronic Wigner correction, then you also likely should not use the population window, but perhaps can use an energy window of ~1 eV around eFermi. See Cepellotti and Kozinsky, Materials Today Physics 19, 100412 Fig. 4 and the related discussion on this as to why this is needed -- if using this feature, we recommend you converge the calculation with respect to window size. 

