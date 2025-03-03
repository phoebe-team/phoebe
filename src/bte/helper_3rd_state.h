#ifndef HELPER_3RD_STATE_H
#define HELPER_3RD_STATE_H

#include "phonon_h0.h"
#include "vector_bte.h"

/** This is a highly specialized auxiliary class, whose sole purpose is to
 * optimize the construction of the phonon scattering matrix.
 * It is used to efficiently return the harmonic information (energy, velocity,
 * and Bose--Einstein distribution) of the third states in the phonon
 * scattering, i.e. at the points q3 = q1 + q2 and q3 = q1-q2.
 *
 * This class handle three cases.
 * 1) If, in the scattering matrix, inner=outer band structure, and the mesh on
 * which the band structure is computed is a FullPoints class, then, the
 * q3-band structure = inner = outer band structure. In this case, Helper3rdState
 * returns information from the innerBandStructure without need for recomputing
 * anything.
 * 2) If, in the scattering matrix, inner=outer band structure, but the mesh on
 * which the band structure is computed is not a FullPoints class (e.g. after we
 * applied a window), then, some values of q3 may not fall on the inner or
 * outer band structure. In this case, we recompute the q3-band structure for all
 * the values that will be needed in the calculation.
 * 3) If, in the scattering matrix, inner is not equal to the outer
 * band structure, q3 is a generic point. Since storing all values of q3 might
 * take too much memory, we proceed in this way. Since the phonon scattering
 * matrix has an outer loop over q2, we compute at once all the band structure
 * for the values of all q3 such that q3 = q1 +- q2 for a fixed q2. Then, we
 * can quickly retrieve the info at each value of q3.
 */
class Helper3rdState {
 public:

  /** Default constructor.
   *
   * The constructor recognize which of the 3 cases we are in.
   * Here we link the 3rd point band structure if we are in case 1
   * (storedAllQ3Case==1 && storedAllQ3), or we compute the band structure if we
   * are in case 2 (storedAllQ3Case==2 && storedAllQ3).
   * We note down if we are in case 3 (!storedAllQ3).
   */
  Helper3rdState(BaseBandStructure &innerBandStructure_,
                 BaseBandStructure &outerBandStructure_,
                 Eigen::MatrixXd &outerBose_,
                 StatisticsSweep &statisticsSweep_,
                 const int &smearingType_,
                 PhononH0& h0_); // = nullptr);

  /** to be called inside the loop over q2, but outside the loop on q1.
   * The order is important, and we assume that the scattering matrix is
   * computed as:
   * for ( iq2 ) {
   *   for ( iq1 ) {
   *     ...
   *
   * If we have 3rd band structure already computed for all wavevectors (cases
   * 1 and 2) we don't do anything. Otherwise, we compute all q3 for this list
   * of q1 (case 3).
   */
  void prepare(const std::vector<int>& q1Indexes, const int &iq2);

  /** to be called inside the loops on q1 and q2 to get the harmonic info on q3.
   */
  std::tuple<Eigen::Vector3d, Eigen::VectorXd, int, Eigen::MatrixXcd,
             Eigen::MatrixXd, Eigen::MatrixXd> get(Point &point1,
                                                   Point &point2,
                                                   const int &thisCase);

  /** To be used with get(), this identifies q3 as q3 = q1 + q2
   */
  static const int casePlus;

  /** To be used with get(), this identifies q3 as q3 = q1 - q2
   */
  static const int caseMinus;

 private:
  BaseBandStructure &innerBandStructure;
  BaseBandStructure &outerBandStructure;
  Eigen::MatrixXd &outerBose;
  StatisticsSweep &statisticsSweep;
  int smearingType;
  PhononH0& h0; // = nullptr;

  std::unique_ptr<BaseBandStructure> bandStructure3;
  std::unique_ptr<Points> fullPoints3;

  bool storedAllQ3; // if true, q3 falls on a grid and we store the full bands

  const int storedAllQ3Case1=1;
  const int storedAllQ3Case2=2;
  int storedAllQ3Case;
  int cacheOffset = 0;
  int numCalculations;

  std::vector<Eigen::VectorXd> cachePlusEnergies, cacheMinusEnergies;
  std::vector<Eigen::MatrixXcd> cachePlusEigenVectors, cacheMinusEigenVectors;
  std::vector<Eigen::MatrixXd> cachePlusBose, cacheMinusBose;
  std::vector<Eigen::MatrixXd> cachePlusVelocity, cacheMinusVelocity;

};

#endif
