
Electron BTE
============

Introduction to the BTE
-------------------------------

Let :math:`f_{\lambda}` be the out-of-equilibrium electron occupation number, where :math:`\nu = (\boldsymbol{k},b)` labels both electronic wavevectors and band index (i.e. the single-particle Bloch numbers).
First, we rewrite the occupation number as:

.. math::
   f_{\lambda} = \bar{f}_{\lambda} + \delta f_{\lambda}

where :math:`\bar{f}_{\lambda}` is the Fermi--Dirac distribution function and we introduced :math:`\delta f_{\lambda}` as the canonical distribution function.

Following a process similar to what was done in phonon BTE section, the linearized electronic BTE for a system exposed to an applied electric field and thermal gradient can be written as,

.. math::
   e \boldsymbol{v}_{\lambda} \cdot \boldsymbol{E} \frac{\partial \bar{f}_{\lambda}}{\partial \epsilon} + \boldsymbol{v}_{\lambda} \cdot \boldsymbol{\nabla} T \frac{\partial \bar{f}_{\lambda}}{\partial T} =
     - \sum_{\lambda'} \Omega_{\lambda\lambda'} \delta f_{\lambda'}

where the first term describes the diffusion due to an externally applied electric field :math:`\boldsymbol{E}`, the second  the diffusion due to a temperature gradient, and the third term is the linearized scattering operator.

The electron scattering matrix :math:`\Omega_{\lambda,\lambda'}` can be computed as

.. math::
   \Omega_{\boldsymbol{k}b,\boldsymbol{k}'b'} =&
   \frac{1}{\tau_{kb}} \delta_{kb,k'b'} + (1-\delta_{kb,k'b'})
   \frac{2\pi}{N_k\hbar} \sum_{\boldsymbol{q}\nu}
   |g_{bb'\nu}(\boldsymbol{k},\boldsymbol{k}')|^2 \\
   &\times
   \bigg[
   ( 1 - \bar{f}_{\boldsymbol{k}b} + \bar{n}_{\boldsymbol{q}\nu})
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} + \hbar \omega_{\boldsymbol{q}\nu}) \\
   &+
   (\bar{f}_{\boldsymbol{k}b} + \bar{n}_{\boldsymbol{q}\nu})
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} - \hbar \omega_{\boldsymbol{q}\nu})
   \bigg]
   \delta(\boldsymbol{k}-\boldsymbol{k}'+\boldsymbol{q}),

with

.. math::
   \frac{1}{\tau_{kb}} =&
   \frac{2\pi}{N_k\hbar} \sum_{b'\boldsymbol{k}',\nu \boldsymbol{q}}
   |g_{bb'\nu}(\boldsymbol{k},\boldsymbol{k}')|^2
   \times
   \bigg[
   (1-\bar{f}_{\boldsymbol{k}'b'} + \bar{n}_{\boldsymbol{q}\nu})
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} - \hbar \omega_{\boldsymbol{q}\nu}) \\ 
   &+
   (\bar{f}_{\boldsymbol{k}'b'} + \bar{n}_{\boldsymbol{q}\nu})
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} + \hbar \omega_{\boldsymbol{q}\nu})
   \bigg]
   \delta(\boldsymbol{k}-\boldsymbol{k}'+\boldsymbol{q}). 
   
This scattering matrix requires us to know the phonon and electron energies, as well as the electron-phonon coupling on a fine (interpolated) mesh.

Please note that, for convenience, here we use a coupling defined as

.. math::
   g_{bb'\nu}(\boldsymbol{k},\boldsymbol{k}')
   =
   g_{b'b\nu}(\boldsymbol{k},\boldsymbol{q})

where the latter can be interpolated as described above.

The Dirac delta function conserving momentum is enforced exactly, since we are using points on a uniform grid centered at gamma.
The Dirac delta conserving energy is instead with a Gaussian function, as described in the section :ref:`delta_fns`.

Since the scattering matrix :math:`\Omega_{\lambda,\lambda'}` is not symmetric, we instead perform the transformation:

.. math::
   \tilde{\Omega}_{\lambda \lambda'}
   =
   \Omega_{\lambda \lambda'}
   \sqrt{ \frac{\bar{f}_{\lambda'} (1-\bar{f}_{\lambda'})}{\bar{f}_{\lambda} (1-\bar{f}_{\lambda})} }

which results in the matrix with diagonal matrix elements:

.. math::
   \tilde{\Omega}_{\lambda \lambda}
   =
   \Omega_{\lambda \lambda} = \frac{1}{\tau_{\boldsymbol{k}b}}

and for the off-diagonal terms:
   
.. math::
   \tilde{\Omega}_{\boldsymbol{k}b,\boldsymbol{k}'b'} =&
   -
   \frac{2\pi}{V N_k} \sum_{\boldsymbol{q}\nu}
   |g_{bb'\nu}(\boldsymbol{k},\boldsymbol{k}')|^2 \delta(\boldsymbol{k}-\boldsymbol{k}'+\boldsymbol{q}) \\
   &\times
   \bigg[
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} + \hbar \omega_{\boldsymbol{q}\nu}) +
   \delta(\epsilon_{\boldsymbol{k}b} - \epsilon_{\boldsymbol{k}'b'} - \hbar \omega_{\boldsymbol{q}\nu})
   \bigg]
   \frac{1}{2 \sinh{\big( \frac{\hbar \omega_{\boldsymbol{q}\nu}}{2 k_B T} \big)  } }
   .

This matrix is symmetric and has a number of interesting physical properties (e.g. the eigenvalues corresponding to the exact relaxation times of the electronic bath).
Computationally, the symmetric matrix can be used in a conjugate gradient method that maximises the electrical and thermal conductivity, and guarantees the existence of eigenvalues.

In Phoebe, instead of solving the original BTE problem in the form :math:`\sum_{\lambda'} \Omega_{\lambda,\lambda'} \delta f_{\lambda'} = b_{\lambda}`, we solve the symmetrized problem:
   
.. math::
   \sum_{\lambda'} \tilde{\Omega}_{\lambda,\lambda'} \delta \tilde{f}_{\lambda'} = \tilde{b}_{\lambda}

with 

.. math::
   \delta f_{\lambda} = ( \bar{f}_{\lambda} (1-\bar{f}_{\lambda}) )^{-\frac{1}{2}} \delta f_{\lambda}

and

.. math::
   \tilde{b}_{\lambda} = ( \bar{f}_{\lambda} (1-\bar{f}_{\lambda}) )^{-\frac{1}{2}} b_{\lambda}


   

Onsager coefficients
--------------------

In the electronic case, there are a handful of transport coefficients we'd like to solve the BTE to find, such as the electrical conductivity, :math:`\sigma`, the electronic part of the thermal conductivity, :math:`\kappa_e`, and the Seebeck coeffieint, :math:`S`. These quantities are defined in terms of the Onsager coefficients. 

We assume that the response to the applied electric field and thermal gradient is linear in these external fields:

.. math::
   \delta f_{\lambda} = \sum_{i} \delta^i f^E_{\lambda} E_i + \delta^i f^T_{\lambda} \nabla_i T


After computing the out-of-equilibrium population, the charge and heat flux density can be computed as:

.. math::
   \boldsymbol{J} = \frac{e g_s}{V N_k} \sum_{\lambda} \boldsymbol{v}_{\lambda} f_{\lambda}

and

.. math::
   \boldsymbol{Q} = \frac{g_s}{V N_k} \sum_{\lambda} (\epsilon_{\lambda}-\mu) \boldsymbol{v}_{\lambda} f_{\lambda}

where :math:`g_s` is the spin degeneracy.

We can decompose these to write, 

.. math::
   \boldsymbol{J} = L_{EE} \boldsymbol{E} + L_{ET} \boldsymbol{\nabla} T

.. math::
   \boldsymbol{Q} = L_{TE} \boldsymbol{E} + L_{TT} \boldsymbol{\nabla} T


From this, the electrical conductivity :math:`\sigma`, the thermal conductivity :math:`k`, the Seebeck coefficient :math:`S` and the mobility :math:`\mu` are:

.. math::
   \sigma = L_{EE}

.. math::
   k = L_{TT} - L_{TE} L_{EE}^{-1} L_{ET}

.. math::
   S = - L_{EE}^{-1} L_{ET}

.. math::
   \mu = \frac{\sigma}{d}

where :math:`d` is the carriers' doping concentration.


Solutions of the electron BTE
--------------------------------------

Largely, these solvers follow the equivalent section in the phonon BTE section, where they may be described in more detail. For further details and references on any specific solver, we suggest you visit the equivalent phonon sections, as well. Here, we again establish methods of finding the solution vector to the BTE, :math:`f`, but in this case, we have two: :math:`f^T` and :math:`f^E`, for each field. 


RTA Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this simple level of theory, we define the electron lifetime as:

.. math::
   A_{ \boldsymbol{k}b,\boldsymbol{k}b } = \frac{1}{ \tau_{\boldsymbol{k}b} }

Next, we approximate the scattering matrix as diagonal, so that the BTE becomes:

.. math::
   e \boldsymbol{v}_{\lambda} \cdot \boldsymbol{E} \frac{\partial \bar{f}_{\lambda}}{\partial \epsilon} + \boldsymbol{v}_{\lambda} \cdot \boldsymbol{\nabla} T \frac{\partial \bar{f}_{\lambda}}{\partial T} =
     - \frac{1}{ \tau_{\lambda} } \delta f_{\lambda}

Solving separately for the response to the electric field and the thermal gradient, we find,

.. math::
   \delta^i f^E_{\lambda} = - e v^i_{\lambda} \frac{\bar{f}_{\lambda}(1-\bar{f}_{\lambda})}{k_B T} \tau_{\lambda}

.. math::
   \delta^i f^T_{\lambda} = - v^i_{\lambda} \frac{(\epsilon_{\lambda}-\mu)\bar{f}_{\lambda}(1-\bar{f}_{\lambda})}{k_B T^2} \tau_{\lambda}




Iterative solution: Omini-Sparavigna method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Generally, we recommend the variational method over this.

This is an adaptation of the Omini-Sparavigna method to electrons. To better understand this method, please have a look first at the counterpart phonon section. 

In short, the electron BTE consists in two linear algebra problems:

.. math::
   m^{i}_{\lambda} = - \sum_{\lambda'} A_{\lambda\lambda'} \delta f_{\lambda}^E


.. math::
   n^{i}_{\lambda} = - \sum_{\lambda'} A_{\lambda\lambda'} \delta f_{\lambda}^T

where

.. math::
   m^{i}_{\lambda} = e v_{\lambda}^i \frac{\partial \bar{f}_{\lambda}}{\partial \epsilon}

.. math::
   n^{i}_{\lambda} = v_{\lambda}^i \frac{\partial \bar{f}_{\lambda}}{\partial T}

The iterative scheme solves these two independent linear algebra problems with a geometric series,

.. math::
   \delta^i f^E_{K} = \sum_{K} \left(-\frac{1}{\boldsymbol{A}^{\mathrm{out}}}  \boldsymbol{A}^{\mathrm{in}}\right)^{K} \frac{1}{\boldsymbol{A}^{\mathrm{out}}} \:  m^i

and

.. math::
   \delta^i f^T_K = \sum_{K} \left(-\frac{1}{\boldsymbol{A}^{\mathrm{out}}}  \boldsymbol{A}^{\mathrm{in}}\right)^{K} \frac{1}{\boldsymbol{A}^{\mathrm{out}}} \:  n^i

where :math:`K` is an iteration index, :math:`A^{in}` is the off-diagonal part of the scattering matrix, and :math:`A^{out}` is the diagonal part of the scattering matrix.
In the code, the two problems are solved together, as we compute the action on the two different vectors at the same time.

Note that, like any geometric series, this algorithm may not converge.

Iterative solution: Variational method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, this solver is very similar to the phonon case (and we recommend you read more there as well).
The only difference for electronic systems is that we need to solve two problems simultaneously, one for the electric field response and one for the response to the thermal gradient.

For the variational method, we can define the variational thermal conductivity, in closed-circuit conditions, as:

.. math::
   k^\mathrm{V}(\delta f^T) = - 2 \mathcal{T}({\delta f^T})

where

.. math::
   \mathcal{T}(\delta f^T) = \frac{1}{2} \sum_{\lambda \lambda'} {\delta f^T_{\lambda}} \cdot{\boldsymbol A_{\lambda\lambda'}} {\delta f^T_{\lambda'}} - \sum_{\lambda} {\boldsymbol n_{\lambda}} \cdot {\delta f^T_{\lambda}}

The variational electrical conductivity is defined similarly as:

.. math::
   \sigma^\mathrm{V}(\delta f^E) = 2 \mathcal{E}({\delta f^E})

where

.. math::
   \mathcal{E}(\delta f^E) = \frac{1}{2} \sum_{\lambda \lambda'} {\delta f^E_{\lambda}} \cdot{\boldsymbol A_{\lambda\lambda'}} {\delta f^E_{\lambda'}} - \sum_{\lambda} {\boldsymbol m_{\lambda}} \cdot {\delta f^E_{\lambda}}


These two functionals are the minimization targets of a conjugate gradient method.
Knowing this, the variational method is exactly the same as the phonon case, with the proper substitution of the vector `b` with either :math:`m` or :math:`n`.

As in the case of the Omini-Sparavigna method, we solve the two equations (response to electric field and thermal gradient) at the same time, as it allows us to minimize the number of times the scattering matrix is evaluated (the most expensive step).



Relaxons solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As for the phonon case, in this scheme, we use an algebraic solution to the BTE, solving the equation in the eigenvector basis.
We first diagonalize the scattering matrix,

.. math::
   \frac{1}{N_k} \sum_{\lambda'} A_{\lambda\lambda'} \theta_{\lambda'\alpha} = \frac{1}{\tau_{\alpha}} \theta_{\lambda\alpha}

where :math:`\theta` are eigenvectors, :math:`\alpha` are eigenvalue indices, and :math:`\frac{1}{\tau_{\alpha}}` are eigenvalues.
We first build the auxiliary quantities:

.. math::
   \delta^i f^E_{\alpha} = \sum_{\lambda} \frac{\partial \bar{f}_{\lambda}}{\partial \epsilon} v_{\lambda}^i  \theta_{\lambda \alpha} \tau_{\alpha}

.. math::
   \delta^i f^T_{\alpha} = \sum_{\lambda} \frac{\partial \bar{f}_{\lambda}}{\partial T} v_{\lambda}^i  \theta_{\lambda \alpha} \tau_{\alpha}

From these, we can compute the solutions of the BTE as:

.. math::
   \delta f^E_{\lambda} = \frac{1}{N_k V} \sum_{\alpha} f^E_{\alpha} \theta_{\lambda \alpha}

.. math::
   \delta f^T_{\lambda} = \frac{1}{N_k V} \sum_{\alpha} f^T_{\alpha} \theta_{\lambda \alpha}


Wigner correction to the electron BTE
---------------------------------------

The theory developments for the Wigner corrections to the electron BTE are described in `Materials Today Physics 19, 100412 (2021). <10.1016/j.mtphys.2021.100412>`_

The Wigner transport equation is

.. math::
   \frac{\partial f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)}{\partial t}
   &+
   \frac{i}{\hbar} \Big[ \mathcal{E}(\boldsymbol{k}) + \boldsymbol{D}(\boldsymbol{k})\cdot\boldsymbol{E} , f(\boldsymbol{x},\boldsymbol{k},t) \Big]_{bb'}
   +
   \frac{1}{2} \Big\{ \boldsymbol{v}(\boldsymbol{k}) , \cdot \frac{\partial f(\boldsymbol{x},\boldsymbol{k},t)}{\partial \boldsymbol{x}} \Big \}_{bb'} \\\\
   &+
   e \boldsymbol{E} \cdot \frac{\partial f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)}{\partial \boldsymbol{k}}
   =
   -\frac{\partial f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)}{\partial t} \bigg|_{coll}

where :math:`f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)` is the Wigner distribution function, :math:`{ \cdot,\cdot }` indicates an anticommutator, :math:`[ \cdot,\cdot ]` indicates a commutator, :math:`v_{bb'}(\boldsymbol{k})` is the velocity operator, and we defined the matrix :math:`\mathcal{E}(\boldsymbol{k})_{bb'} = \delta_{bb'} \epsilon_{\boldsymbol{k}b}` and :math:`\mathcal{D}(\boldsymbol{k})_{bb'} = (1-\delta_{bb'}) d_{\boldsymbol{k}bb'}` is a matrix of electronic dipoles.
The electronic dipole can be computed as:

.. math::
   \boldsymbol{d}_{\boldsymbol{k},bb'}
   =
   - i e \frac{\boldsymbol{v}_{bb'}(\boldsymbol{k})}{\epsilon_{b}(\boldsymbol{k})-\epsilon_{b'}(\boldsymbol{k})}  , \quad \text{for }b \neq b'


The scattering operator acts on the diagonal Wigner distribution as the BTE scattering operator, instead it acts on the off-diagonal components with a decay term:

.. math::
   \frac{\partial f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)}{\partial t} \bigg|_{coll}
   =
   (1-\delta_{bb'}) \frac{\Gamma_{b}(\boldsymbol{k}) + \Gamma_{b'}(\boldsymbol{k})}{2} f_{bb'}(\boldsymbol{x},\boldsymbol{k},t)
   +
   \delta_{bb'} \frac{1}{V}
   \sum_{\boldsymbol{k}'b'} A_{\boldsymbol{k}b,\boldsymbol{k}'b'} f_{b'b'}(\boldsymbol{x},\boldsymbol{k}',t)

where :math:`\Gamma_b(\boldsymbol{k}) = \frac{2\pi}{\tau_{\boldsymbol{k}b}}` are the electronic linewidths.

To solve the Wigner transport equation, just like we did for the BTE, we assume linear response and separate the response to electric field and thermal gradient :math:`f = f^E E + f^T \nabla T`.
The diagonal part of the Wigner transport equation is exactly equal to the BTE, and can be solved using one of solvers described above.
The off-diagonal part of the Wigner distribution function can be solved easily with a little algebraic manipulation.

The related transport coefficients are defined as:

.. math::
   L_{EE}^{ij} =
   \frac{e g_s}{V N_k} \sum_{\boldsymbol{k}b} \frac{1}{2} \Big\{ v^i(\boldsymbol{k}) , f^{E_j}(\boldsymbol{k}) \Big\}_{bb}

.. math::
   L_{ET}^{ij} =
   \frac{e g_s}{V N_k} \sum_{\boldsymbol{k}b} \frac{1}{2} \Big\{ v^i(\boldsymbol{k}) , f^{T_j}(\boldsymbol{k}) \Big\}_{bb}

.. math::
   L_{TE}^{ij} =
   \frac{g_s}{V N_k}
   \sum_{\boldsymbol{k}b}
   \big( \epsilon_{b}(\boldsymbol{k})-\mu \big)
   \frac{1}{2} \Big\{ v^i(\boldsymbol{k}) , f^{E_j}(\boldsymbol{k}) \Big\}_{bb}

.. math::
   L_{TT}^{ij} =
   \frac{g_s}{V N_k}
   \sum_{\boldsymbol{k}b}
   \big( \epsilon_{b}(\boldsymbol{k})-\mu \big)
   \frac{1}{2} \Big\{ v^i(\boldsymbol{k}) , f^{T_j}(\boldsymbol{k}) \Big\} _{bb}

