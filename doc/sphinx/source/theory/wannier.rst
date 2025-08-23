Additional Computational Details
=================================

Wannier interpolation of band structure
---------------------------------------

A good review of the Wannier function formalism can be found at this link (https://arxiv.org/abs/1112.5411).

We assume that a third-party code has provided us with the single-particle Hamiltonian in the Wannier representation:

.. math::
   \langle \boldsymbol{0}n | H | \boldsymbol{R} m \rangle

where :math:`\boldsymbol{R}` labels a bravais lattice vector in a supercell, :math:`m` and :math:`n` label two wannier functions.
A Wannier function is here denoted by the ket:

.. math::
   | \boldsymbol{R} m \rangle


The Wannier interpolation procedure requires us to transform from the real-space Wannier representation to reciprocal space (as well as the reverse transform),

.. math::
   H_{\boldsymbol{k},nm}^W = \sum_{\boldsymbol{R}} e^{i \boldsymbol{k} \cdot \boldsymbol{R}} \langle \boldsymbol{0}n | H | \boldsymbol{R} m \rangle

This sum is typically performed over irreducible lattice vectors:

.. math::
   H_{\boldsymbol{k},nm}^W = \sum_{\boldsymbol{R}_{irr}} \frac{e^{i \boldsymbol{k} \cdot \boldsymbol{R}_{irr}} }{ d_{\boldsymbol{R}_{irr}}} \langle \boldsymbol{0} n | H | \boldsymbol{R} m \rangle

where :math:`d_{\boldsymbol{R}_{irr}}` is the degree of degeneracy of the irreducible bravais lattice vector :math:`\boldsymbol{R}_{irr}`.

The list of :math:`\boldsymbol{R}_{irr}` vectors is built, by default, from all the Bravais lattice vectors lying inside a supercell of the crystal primitive cell; the size of such supercell is equivalent to the size of the grid of :math:`k` wavevectors.
The degeneracy :math:`d_{\boldsymbol{R}_{irr}}` keeps track of the number of equivalent lattice vectors: due to use of periodic boundary conditions, lattice vectors at the faces/sides/corners of the supercell can be up to 8-fold degenerate.
Especially relevant when the grid of wavevectors is small, the accuracy of the Fourier transform can be improved by applying shifts to these lattice vectors.
To this aim, the lattice vectors :math:`\boldsymbol{R}_{irr}` can be shifted, such that :math:`\boldsymbol{R}_{irr}` lies in the supercell centered at the midway point between the Wannier functions :math:`n` and :math:`m`.

The Hamiltonian matrix :math:`H_{\boldsymbol{k},nm}^W` is not diagonal, as we are working in the maximally-localized Wannier function gauge.
We thus diagonalize the matrix to pass to the Bloch representation:

.. math::
   H_{\boldsymbol{k},bb'}^B = [U_{\boldsymbol{k}}^\dagger H_{\boldsymbol{k}}^W U_{\boldsymbol{k}}]_{bb'} = \delta_{bb'} \epsilon_{\boldsymbol{k}b}

Therefore finding the interpolated electronic energies at a generic :math:`\boldsymbol{k}` point.

The velocity operator is computed with the Hellmann-Feynman theorem, e.g. along the x direction as:

.. math::
   v^x_{\boldsymbol{k}bb'} = [U_{\boldsymbol{k}}^\dagger \frac{H_{\boldsymbol{k}+\boldsymbol{\delta}_x}^W-H_{\boldsymbol{k}-\boldsymbol{\delta}_x}^W}{2 \delta_x} U_{\boldsymbol{k}}]_{bb'}

Whenever we find a set of degenerate bands at a given k point, we diagonalize the velocity operator in the degenerate subset, in order to uniquely define the velocity operator.

.. _theoryElphWannier:

Wannier interpolation of electron-phonon coupling
-------------------------------------------------

This is a summary of the strategy for interpolation adopted by Phoebe, as detailed also at this reference (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.165108).

The coupling to be used for the calculation of scattering rates is:

.. math::
   g^{SE}_{b'b,\nu} (\boldsymbol{k},\boldsymbol{q}) = \bigg( \frac{1}{2 m \omega_{\boldsymbol{q}\nu}} \bigg)^{1/2} g_{b'b,\nu} (\boldsymbol{k},\boldsymbol{q})


What we need to interpolate instead is:

.. math::
   g_{b'b,\nu} (\boldsymbol{k},\boldsymbol{q}) = \big<b'\boldsymbol{k}+\boldsymbol{q} \big| \partial_{\boldsymbol{q}\nu}V \big| b\boldsymbol{k} \big>


First, recall the relation between the wavefunctions in Wannier and Bloch representations,

.. math::
   \big|m\boldsymbol{R}_e\big> = \sum_{b\boldsymbol{k}} e^{-i\boldsymbol{k}\cdot\boldsymbol{R}_e} U_{mb,\boldsymbol{k}} \big|b\boldsymbol{k}\big>


.. math::
   \big|b\boldsymbol{k}\big> = \frac{1}{N_e} \sum_{m\boldsymbol{R}_e} e^{i\boldsymbol{k}\cdot\boldsymbol{R}_e} U_{bm,\boldsymbol{k}}^\dagger \big|m\boldsymbol{R}_e\big>

where :math:`N_e` is the number of supercells.



Let :math:`e_{\boldsymbol{q}\kappa}^{\nu}` be the phonon eigenvector we get from diagonalizing the dynamical matrix.
We define :math:`u_{\boldsymbol{q}\kappa}^{\nu} = (\frac{m_0}{m_{\kappa}})^{1/2} e_{\boldsymbol{q}\kappa}^{\nu}`, where :math:`m_0` is the electron mass, and :math:`m_{\kappa}` is the ionic mass.

To transform the potential from the reciprocal to the real space representation, we have:

.. math::
   \partial_{\kappa \boldsymbol{R}_p} V(\boldsymbol{r})
   =
   \frac{1}{N_q}
   \sum_{\boldsymbol{q}\nu} e^{-i\boldsymbol{q}\cdot\boldsymbol{R}_p} [u_{\boldsymbol{q}\kappa}^{\nu}]^{-1} \partial_{\boldsymbol{q}\nu} V(\boldsymbol{r})



So, we first transform to Wannier space by:

.. math::
   g(\boldsymbol{R}_e,\boldsymbol{R}_p)
   =
   \frac{1}{N_q N_k}
   \sum_{\boldsymbol{k}\boldsymbol{q}} e^{-i\boldsymbol{k}\cdot\boldsymbol{R}_e-i\boldsymbol{q}\cdot\boldsymbol{R}_p} U_{\boldsymbol{k}+\boldsymbol{q}}^\dagger g(\boldsymbol{k},\boldsymbol{q}) U_{\boldsymbol{k}} u_{\boldsymbol{q}}^{-1}


Then, we interpolate to Bloch space

.. math::
   g(\boldsymbol{k},\boldsymbol{q})
   =
   \frac{1}{N_e}
   \sum_{\boldsymbol{R}_e \boldsymbol{R}_p} e^{i\boldsymbol{k}\cdot\boldsymbol{R}_e+i\boldsymbol{q}\cdot\boldsymbol{R}_p} U_{\boldsymbol{k}+\boldsymbol{q}} g(\boldsymbol{R}_e,\boldsymbol{R}_p) U_{\boldsymbol{k}}^\dagger u_{\boldsymbol{q}}



Details:

* the mesh of :math:`\boldsymbol{k}` and :math:`\boldsymbol{q}` points must be the same (or at least commensurate, so that we can map :math:`\boldsymbol{k}+\boldsymbol{q}` into the same :math:`\boldsymbol{k}` grid).






Computational details, Wannier90
--------------------------------

First, the mesh of :math:`\boldsymbol{k}` and :math:`\boldsymbol{q}` points must be commensurate.
In fact, the matrices :math:`U` are computed on a grid of :math:`\boldsymbol{k}` points.
It is thus necessary that :math:`\boldsymbol{k}+\boldsymbol{q}` falls on the same grid of points.


To be precise, note that we use two different sets of :math:`U` matrices.
When interpolating from the Wannier to the Bloch space, we use the U matrices computed at an arbitrary grid of points obtained by diagonalizing the Hamiltonian matrix in the Wannier space.
When first transforming from the Bloch space (the coupling with an ab-initio code) to the Wannier space, we use the :math:`U` matrices computed by Wannier90.
This is necessary, because the coupling :math:`g` is a complex function, and we must rotate it using the same wavefunction gauge used to build the maximally-localized Wannier functions.
Moreover, Wannier90 may use the disentanglement procedure.
In that case, the Bloch to Wannier transformation is:

.. math::
   \big|m\boldsymbol{R}_e\big> = \sum_{\boldsymbol{k} b b'} e^{-i\boldsymbol{k}\cdot\boldsymbol{R}_e} U^{rot}_{mb',\boldsymbol{k}} U^{dis}_{b'b,\boldsymbol{k}} \big|b\boldsymbol{k}\big>

where the number of disentangled bands :math:`b'` is smaller than the number of entangled bands :math:`b`.
Therefore, we rotate the electron-phonon coupling from the Bloch to Wannier space using the entangled number of bands.
Wannier90 prints the two different :math:`U` matrices, and one can just multiply them to get the transformation matrix.

As a further minor detail, remember that some bands (like deep core bands) may be excluded from the Wannierization procedure (through the keyword exclude-indices), so that there may be an offset in the band index of U and g.



Computational details, gauge fixing in Quantum ESPRESSO
-------------------------------------------------------

The interpolation procedure described above implicitely assumes that the wavefunction :math:`\big|b\boldsymbol{k}\big>` has a fixed gauge.
In fact, all the quantities above are complex numbers, and the wavefunction is defined within a phase (or, more generally, a unitary rotation).
In order for the interpolation to work, we must make sure that the wavefunction used for computing all the quantities above are exactly the same coefficient-wise, phase included, and make sure that pw.x, ph.x and wannier90.x operate on the same wavefunctions.

The problem comes from the arbitrariness of the phase choice of an eigenvector of a Hermitian matrix.
In details: let :math:`H_{\boldsymbol{k}}` be a Bloch Hamiltonian.
The DFT code will diagonalize the Hamiltonian and solve :math:`H_{\boldsymbol{k}} \psi_{\boldsymbol{k}} = \epsilon_k \psi_{\boldsymbol{k}}`.
For each eigenvector :math:`\psi_{\boldsymbol{k}}`, we can apply the transformation :math:`\psi_{\boldsymbol{k}} \to e^{i \theta_{\boldsymbol{k}}} \psi_{\boldsymbol{k}}` and still have :math:`e^{i \theta_{\boldsymbol{k}}} \psi_{\boldsymbol{k}}` an eigenvector.
Note also that the diagonalization may not have a strategy to fix the phase of the eigenvector: as a result, we may expect that every different run of a DFT code will generate a different phase, effectively behaving as a random number generator.

We thus patch the Quantum ESPRESSO code to fix a gauge of the wavefunction.
Additionally, we want to make sure that the wavefunction satisfies rotational symmetries, as this will help us reduce the number of calculations of the electron-phonon coupling at the DFT level.

In a plane-wave code, the wavefunction is expanded in a plane-wave basis set as

.. math::
   \psi_{\boldsymbol{k}} = \sum_{\boldsymbol{G}} c(\boldsymbol{G}) e^{i\boldsymbol{k}\cdot\boldsymbol{G}+i\boldsymbol{k}\cdot\boldsymbol{r}}

Quantum ESPRESSO, stores the plane wave coefficients in :math:`evc(ig,ib)`, where :math:`ib` is a band index and :math:`ig` is a G-vector index.
Details are described in the source code, but keep in mind that :math:`evc` is parallel-distributed over G-vectors, and that each k-point has a different order of G-vectors.
If we want to fix the gauge, we must operate on the plane wave coefficients.

The wavefunction satisfies some symmetries.
Let :math:`S` be a symmetry operation of the crystal.
A symmetry operation consists of a rotation :math:`R` and a fractional translation :math:`t`, that leave the crystal invariant.
As the wavefunction must transform like the crystal, it can be shown that :math:`\psi_{R\boldsymbol{k}}(\boldsymbol{r}) = \psi_{\boldsymbol{k}}(R^{-1}(\boldsymbol{r}-\boldsymbol{t}))`.
From this symmetry property, one can verify that the following relations hold:

.. math::
   \epsilon_{R\boldsymbol{k},n} = \epsilon_{\boldsymbol{k}n}

.. math::
   c_{R\boldsymbol{k},n}(\boldsymbol{G}) = e^{-i(R\boldsymbol{k}+\boldsymbol{G}) \cdot \boldsymbol{t}} c_{\boldsymbol{k}n}(R^{-1}\boldsymbol{G})

Additionally, the wavefunction is periodic over the Brillouin zone, i.e. :math:`\psi_{k}(r) = \psi_{k+G'}(r)`.
From this, it follows that:

.. math::
   \epsilon_{\boldsymbol{k}+\boldsymbol{K},n} = \epsilon_{\boldsymbol{k}n}

.. math::
   c_{\boldsymbol{k}+\boldsymbol{G}',n}(\boldsymbol{G})
   =
   c_{\boldsymbol{k}n}(\boldsymbol{G}+\boldsymbol{G}')

Note: Abinit has a very well curated section on the symmetries of the wavefunction https://docs.abinit.org/theory/wavefunctions/ .

Before fixing the gauge, we also stress an additional problem: electronic degeneracy.
If two (or more) energy levels are degenerate, the wavefunction is only defined up to a unitary rotation.
In fact, let :math:`i` span the subspace of degenerate eigenvalues.
Then, the wavefunctions can be rotated as :math:`\tilde{\psi}_i = \sum_j U_{ij} \psi_j`, with :math:`U` any unitary matrix.
Therefore, when fixing the gauge, we must also deal with this problem: we must also mix the plane wave coefficients of different degenerate bands.

The algorithm to fix the gauge in Quantum ESPRESSO goes as follows:

* Run a scf calculation using the k-points in the irreducible wedge :math:`\{ k^{irr} \}`,
  setting the number of bands equal to what you want to use in both Wannier90 and ph.x.
  Right after the Hamiltonian is diagonalized at a given k-point (in file `PW/src/c_bands.f90`),
  and fix the gauge of non-degenerate eigenvectors by setting c(G=0) to be real and positive.
  For degenerate eigenvalues, set c(G=0)>0 only for the first band of the degenerate subspace.
  Save the wavefunction and its G-vectors (the arrays `evc`, `g_vectors`, and the mapping `igk_k`).

* During a ph.x calculation, or a nscf calculation before Wannier90, the codes ask to
  diagonalize the Hamiltonian at a point k (or k+q) that is commensurate with the grid of points
  used in the scf calculation.
  Given a point k, do:

  * find the irreducible point :math:`k^*` that is symmetry-equivalent to the current point.
    If not found, block the code (the user has either messed symmetries or used wrong k/q meshes).
    Find also the symmetry operation S such that :math:`R k = k^* + K`,
    where :math:`K` is an Umklapp vector.

  * Read the wavefunction at :math:`k^*`.

  * Build `gmap`, a map between indices of two arrays of G-vectors such that
    :math:`G[i] = (R^{-1}G+K)[gmap(i)]`. This will help us apply the roto-translational symmetry.

  * Compute the roto-translated wavefunction :math:`\psi_{Rk} = \psi_{k^*+K}`
    using the relations on the plane-wave coefficients described above.

This would be enough, if the wavefunctions were exact.
Unfortunately, this procedure doesn't allow us to reconstruct the complete wavefunction.
In fact, the wavefunctions are typically expanded over a set of G-vectors such that :math:`|k+G|^2<E_{cut}`.
Therefore, the wavefunction can only be rotated for the intersecting set of G-vectors between the wavefunctions at the irreducible (reference) point and the roto-translated point.
We wouldn't have information for G-vectors outside this intersection and we would set them to zero, breaking the normalization condition.

We bypass this problem in this way.
Let :math:`\big| \psi^{QE} \big>` be the wavefunction computed by QE at point k and :math:`\big| \psi^{rot} \big>` the wavefunction we computed using the roto-translation of the irreducible point.

* Using the relation

.. math::
   \big| \psi^{rot} \big>
   =
   \sum_{QE} \big< \psi^{rot} \big| \psi^{QE} \big>^* \big| \psi^{QE} \big>
   =
   U \big| \psi^{QE} \big>

to define a unitary matrix :math:`U`.

* On paper, :math:`U` should be unitary, i.e. :math:`U U^{\dagger} = 1`.
  But for the same problems of completeness of G-sphere, we have :math:`U U^{\dagger} = 1-\Delta`.
  With some manipulations,

.. math::
   1 = U U^{\dagger} + \Delta = U U^{\dagger} + U U^{\dagger} \Delta U U^{\dagger}
   = U ( 1 + U^{\dagger} \Delta U ) U^{\dagger}
   = U L L^{\dagger} U^{\dagger}

where :math:`L` comes from the Cholesky decomposition of
  :math:`( 1 + U^{\dagger} \Delta U ) = LL^{\dagger}`.

* Redefine :math:`\tilde{U} = UL` (this matrix is unitary by construction).
  Finally, the wavefunction at the point k is :math:`\tilde{U} \big| \psi^{QE} \big>`

This procedure has been implemented in QE, in the file `c_bands.f90`.

Note that there is a catch for entangled bands.
In building the unitary matrix :math:`U`, we assumed completeness of the wavefunction set.
If you are Wannierizing disentangled bands, this is fine.
If you are trying to disentangle some bands, than it is possible that, by choosing the number of bands to be computed, we may cut through a group of degenerate bands.
If this happens, the last block of the matrix :math:`U` may not be unitary, not just because of numerical noise, but because of breaking the completeness relation.
We checked that, as long as you are discarding such bands in the disentangling procedure, the Wannierized wavefunctions should be fine.

Final comments:

1. In order to rotate the wavefunction, each MPI process needs to have enough memory to store
   the complete wavefunction (all G vectors) for a single band,
   i.e., each MPI process requires an additional :math:`16 N_G` Bytes of memory.

2. The lack of completeness implies that, as for any DFT calculation,
   one must converge the G-vectors cutoff (`ecutwfc` in QE).

3. The wavefunction, or g, even though it obeys symmetries,
   it isn't smooth with respect to :math:`\boldsymbol{k}`.
   This is guaranteed by the maximally localized Wannier gauge
   (which in the reciprocal space guarantees continuity with respect to k).

4. Currently we don't support spin, but we will add it soon (must include a few more symmetries).






Computational details, symmetries in Quantum ESPRESSO
-----------------------------------------------------

The phonon code can be used to compute the coupling :math:`g(\boldsymbol{k},\boldsymbol{q})`, where k falls on a Monkhorst-Pack grid of points (nk1,nk2,nk3) and q falls on a Monkhorst-Pack mesh (nq1,nq2,nq3).
We require that both meshes are centered at the Gamma point, so that we have the Wannier90 matrices for the Bloch to Wannier rotation.
Given that the calculation is quite expensive, Quantum ESPRESSO uses symmetries to reduce the required amount of calculations.

As discussed above, we made sure that the set of wavefunctions obeys the relations: :math:`\psi_{R\boldsymbol{k}}(\boldsymbol{r}) = \psi_{\boldsymbol{k}}(R^{-1}(\boldsymbol{r}-\boldsymbol{t}))`.

Intuitively, the electron-phonon coupling itself should remain invariant under a symmetry operation: :math:`g(\boldsymbol{k},\boldsymbol{q}) = g(S\boldsymbol{k},S\boldsymbol{q})` and therefore, it should obey :math:`g(S^{-1}\boldsymbol{k},\boldsymbol{q}) = g(\boldsymbol{k},S\boldsymbol{q})`. More systematically:

.. math::
   g(\boldsymbol{k},S\boldsymbol{q})
   = \big< \psi_{\boldsymbol{}k+S\boldsymbol{q}}(\boldsymbol{r}) \big| \delta V_{S\boldsymbol{q}}(\boldsymbol{r}) \big| \psi_{\boldsymbol{k}}(\boldsymbol{r}) \big> \\\\
   = \big< \psi_{\boldsymbol{k}+S\boldsymbol{q}}(\boldsymbol{r}) \big| \delta V_{\boldsymbol{q}}(S^{-1}\boldsymbol{r}) \big| \psi_{\boldsymbol{k}}(\boldsymbol{r}) \big> \\\\
   = \big< \psi_{\boldsymbol{k}+S\boldsymbol{q}}(S\boldsymbol{r}) \big| \delta V_{\boldsymbol{q}}(\boldsymbol{r}) \big| \psi_{\boldsymbol{k}}(S\boldsymbol{r}) \big> \\\\
   = \big< \psi_{S^{-1}\boldsymbol{k}+\boldsymbol{q}}(\boldsymbol{r}) \big| \delta V_{\boldsymbol{q}}(\boldsymbol{r}) \big| \psi_{S^{-1}\boldsymbol{k}}(\boldsymbol{r}) \big> \\\\
   = g(S^{-1}\boldsymbol{k},\boldsymbol{q})


Note two things: if the wavefunction doesn't rotate with the symmetries of the crystal (e.g. the gauge has not been fixed and degeneracies are not lifted), there will be phase factors hanging around, and the fourth equality in the expressions above doesn't hold.

Additionally, the translational invariance allows us to use the symmetry

.. math::
   g(\boldsymbol{k},\boldsymbol{q}) = g(\boldsymbol{k}+\boldsymbol{G},\boldsymbol{q}+\boldsymbol{G}') \;,


useful whenever a rotated point falls outside the Brillouin zone and must be folded back with an Umklapp vector :math:`\boldsymbol{G}`.

The code ph.x uses two symmetries to reduce the list of :math:`\boldsymbol{k}` and :math:`\boldsymbol{q}` points.
First of all, ph.x only computes the coupling for the irreducible set of q wavevectors.
As a first guess, one may think that ph.x computes the coupling for all k points falling on a Monkhorst-Pack grid, for every irreducible q point.
However, at fixed irreducible :math:`\boldsymbol{q}` point, we don't need to compute all wavevectors :math:`\boldsymbol{k}`.
In fact, consider a symmetry :math:`S` that sends the irreducible point :math:`q` to a reducible point :math:`R\boldsymbol{q}=\boldsymbol{q}^*` that are both on the Monkhorst-Pack mesh of q-points selected in input to ph.x.
While a wavevector :math:`\boldsymbol{k}` also falls on a Monkhors-Pack mesh, it may be that its rotation :math:`R\boldsymbol{k}` doesn't fall on the k-vector grid.
Therefore, we can discard the k-wavevectors of the grid that don't transform like :math:`\boldsymbol{q}` (for each irreducible q) and set their electron-phonon coupling to zero.
The ph.x code computes the coupling only for the pairs of :math:`\boldsymbol{k}` and :math:`\boldsymbol{q}` wavevectors that obey the same subset of symmetries, which can be rotated with the relations described above.
However, before testing this relation, we impose :math:`\boldsymbol{k}` to fall on a full grid.


Computational details, phonon symmetries
----------------------------------------

We should not forget that also the phonon eigenvectors should satisfy the crystal symmetries when used for the Wannier transformation.
The symmetries of phonons are thoroughly discussed in this reference (https://link.aps.org/doi/10.1103/RevModPhys.40.1), from which we need just Eq. 2.33.
In detail, let the phonon eigenvector be :math:`z_{\mu k j}(q)`, where :math:`k` is an atomic basis index, :math:`\mu` is a cartesian index, :math:`q` is the wavevector, and :math:`j` is the mode index.
If :math:`S` is a symmetry operation of the crystal, the phonon eigenvector rotates as:

.. math::
   \boldsymbol{q}' = S\boldsymbol{q}

.. math::
   \omega_{j}(S\boldsymbol{q}) = \omega_{j}(\boldsymbol{q})

.. math::
   z_{\mu K j}(S\boldsymbol{q}) = \sum_{\alpha} S_{\mu\nu} z_{\nu k j}(\boldsymbol{q}) \exp( i\boldsymbol{k} \cdot (S^{-1} R_{at}(K) - R_{at}(k)) )

where :math:`R_{at}` is the atomic position of an atom in the unit cell.
Furthermore, :math:`K` is the atomic basis index of the atom on which the atom :math:`k` is transformed into upon the symmetry operation (since atoms of the same species are indistinguishable, they can be rotated into a different basis index, provided it's the same atomic species).
