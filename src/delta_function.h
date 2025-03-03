#ifndef DELTA_F_H
#define DELTA_F_H

#include "bandstructure.h"
#include "context.h"
#include "eigen.h"
#include "points.h"

/** Base class for the approximations to the Delta Function.
 * Currently used for density of states calculation or for transport
 * calculations (scattering rates require a dirac-delta).
 */
class DeltaFunction {
public:
  /** Destructor for the DeltaFunction method.
   * Needed to destruct the pointer returned by the factory method.
   */
  virtual ~DeltaFunction();

  /** A smearing factory for loading a specific approximation to the
   * dirac-delta. Returns a pointer. Therefore, it should be deleted
   * afterwards.
   * @param context: object with the user input, containing the choice of
   * smearing.
   * @param BaseBandStructure: the object containing the band structure
   * computed on a mesh of wavevectors in the Brillouin zone.
   */
  static DeltaFunction *smearingFactory(Context &context,
                                        BaseBandStructure &baseBandStructure);

  // These are integers identifying the smearing selection of the user.
  static const int gaussian = 0;
  static const int adaptiveGaussian = 1;
  static const int tetrahedron = 2;
  static const int symAdaptiveGaussian = 3;

  /** Method for obtaining a smearing value (approximating \delta).
   * @param energy: the energy difference of the dirac delta
   * @param velocity optional: a vector of velocity (used for adaptive smearing schemes).
   * @param velocity optional: a vector of velocity (used for sym adaptive smearing scheme).
   * @param velocity optional: a vector of velocity (used for sym adaptive smearing scheme).
   */
  virtual double
  getSmearing(const double &energy,
              const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity2 = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity3 = Eigen::Vector3d::Zero()) = 0;

  /** Method for obtaining a smearing value (approximating \delta).
   * Overloads the previous one, and is used by the tetrahedron method.
   * @param energy: the energy difference of the dirac delta.
   * @param iq: wavevector index.
   * @param ib: band index.
   */
  virtual double getSmearing(const double &energy, StateIndex &is) = 0;

  /** Method to identify which kind of smearing is being used.
   * Returns a int value between gaussian, adaptiveGaussian, tetrahedron.
   */
  virtual int getType();

private:
  int id = -1;
};

/** Gaussian smearing scheme. Simply replace the dirac delta with a gaussian.
 * The width of such gaussian is specified by the user.
 */
class GaussianDeltaFunction : public DeltaFunction {
public:
  /** Constructor of the gaussian smearing scheme.
   * @param context: object with user input.
   */
  explicit GaussianDeltaFunction(BaseBandStructure& bandStructure,Context &context); // context to get amplitude

  /** Method to obtain the value of smearing.
   * @param energy: the energy difference.
   * @param[optional] velocity: ignored parameter.
   * @param[optional] velocity: ignored parameter.
   * @param[optional] velocity: ignored parameter.
   * @return smearing: the approximation to the dirac-delta.
   */
  double getSmearing(const double &energy,
              const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity2 = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity3 = Eigen::Vector3d::Zero()) override;

  /** phantom method that should not be used. Will throw an error.
   */
  double getSmearing(const double &energy, StateIndex &is) override;

  /** returns an integer identifying this class as AdaptiveGaussian
   * @return int: id.
   */
  int getType() override;

protected:
  int id = DeltaFunction::gaussian;
  double inverseWidth;
  double prefactor;
};

/** Adaptive smearing scheme. Uses the group velocity to adjust the width
 * of a gaussian approximation to the dirac-delta.
 * See Ref doi: 10.1103/PhysRevB.75.195121
 */
class AdaptiveGaussianDeltaFunction : public DeltaFunction {
public:
  /** Constructor of the adaptive gaussian smearing scheme.
   * @param bandStructure: mainly to see what mesh of points and crystal is
   * being used, and to prepare a suitable scaling of velocities.
   */
  explicit AdaptiveGaussianDeltaFunction(BaseBandStructure &bandStructure, Context& context,
                                         double broadeningCutoff_=0.0001 / energyRyToEv);

  /** Method to obtain the value of smearing.
   * @param energy: the energy difference.
   * @param velocity; the velocity (difference) of the quasi-particles.
   * @param[optional] velocity: ignored parameter used by sym adaptive.
   * @param[optional] velocity: ignored parameter used by sym adaptive.
   * @return smearing: the approximation to the dirac-delta
   */
  double
  getSmearing(const double &energy,
              const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity2 = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity3 = Eigen::Vector3d::Zero()) override;

  /** Phantom method that should not be used. Will throw an error.
   */
  double getSmearing(const double &energy, StateIndex &is)  override;

  /** returns an integer identifying this class as AdaptiveGaussian
   * @return int: id.
   */
  int getType() override;

protected:
  int id = DeltaFunction::adaptiveGaussian;

  double broadeningCutoff;

  const double prefactor = 1.; // could be exposed to the user
  Eigen::Matrix3d qTensor;     // to normalize by the number of wavevectors.

  // since we truncate the smearing above 2sigma, we use erf to correct
  // results
  const double erf2 = 0.9953222650189527;

  // choose a = 2 (though this is overwritten if adaptivePrefactor is set in input)
  // which seems to remove negative el relaxons eigenvalues at low T
  // looking at eq 33 and 34 here,
  // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.75.195121
  // Apparently expect good good results for 0.8 < a < 1.3
  double adaptivePrefactor = sqrt(1. / 6.);
  bool isPhonon;

};

/** Class for approximating the Delta function with the tetrahedron method
 */
class TetrahedronDeltaFunction : public DeltaFunction {
public:
  int getType() override;

  /** Constructor method.
   * Forms all tetrahedra for the 3D wave vector mesh.
   * Lower dimensionality is not currently supported.
   *
   * Method for creating and enumerating all the tetrahedra
   * for a given 3D mesh of wave vectors following Fig. 5 of
   * Bloechl, Jepsen and Andersen prb 49.23 (1994): 16223.
   *
   * @param[in] bandStructure: the band structure on which the dirac delta
   * will be computed.
   */
  explicit TetrahedronDeltaFunction(BaseBandStructure &fullBandStructure_);

  /** Calculate the total tetrahedron weight for all states at given energy.
   *
   * Method for calculating the tetrahedron weight for given wave vector and
   * polarization following Lambin and Vigneron prb 29.6 (1984): 3430.
   *
   * @param[in] energy Energy of mode.
   * @return dos: density of states for the given value of energy
   */
  double getDOS(const double &energy);

  /** Calculate tetrahedron weight.
   *
   * Method for calculating the tetrahedron weight for given wave vector and
   * polarization following Lambin and Vigneron prb 29.6 (1984): 3430.
   *
   * @param[in] energy Energy of mode.
   * @param[in] State: state at which the tetrahedron is computed.
   * @returns weight: the tetrahedron weight approximating the dirac delta.
   */
  double getSmearing(const double &energy, StateIndex &is) override;

  /** overload abstract base method, but simply raises an error if it's
   * called. One should really use the other getSmearing method.
   */
  double
  getSmearing(const double &energy,
              const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity2 = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity3 = Eigen::Vector3d::Zero()) override;

protected:
  BaseBandStructure &fullBandStructure;
  Points fullPoints;
  int id = DeltaFunction::tetrahedron;
  Eigen::MatrixXd subCellShift;
  Eigen::MatrixXi vertices;
};

/** Adaptive smearing scheme which should respect symmetry of the scattering matrix.
 * Uses the state group velocities to adjust the width of a gaussian approx to the dirac-delta.
 * See Ref doi: 10.1103/PhysRevB.75.195121 (for regular adaptive smearing)
 * and 3.3.1 of Ref https://doi.org/10.1016/j.cpc.2022.108504 for this updated version
 */
class SymAdaptiveGaussianDeltaFunction : public AdaptiveGaussianDeltaFunction {
public:
  /** Constructor of the sym adaptive gaussian smearing scheme.
   * @param bandStructure: mainly to see what mesh of points and crystal is
   * being used, and to prepare a suitable scaling of velocities.
   */
  explicit SymAdaptiveGaussianDeltaFunction(BaseBandStructure &bandStructure, Context& context,
                                         double broadeningCutoff_=0.0001 / energyRyToEv);

  /** Method to obtain the value of smearing.
   * @param energy: the energy difference.
   * @param velocity; the velocity of quasiparticle state 1
   * @param velocity; the velocity of quasiparticle state 2
   * @param velocity; the velocity of quasiparticle state 3
   * @return smearing: the approximation to the dirac-delta
   */
  double getSmearing(const double &energy,
              const Eigen::Vector3d &velocity = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity2 = Eigen::Vector3d::Zero(),
              const Eigen::Vector3d &velocity3 = Eigen::Vector3d::Zero()) override;

protected:
  // Most of this object is inherited from parent adaptive gaussian
  // We set id in the constructor, as we cannot here override an inherited class member
};

#endif
