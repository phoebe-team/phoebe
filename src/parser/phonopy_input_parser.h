#ifndef PHONOPY_PARSER_H
#define PHONOPY_PARSER_H

#include "context.h"
#include "phonon_h0.h"
#include <string>

/** Class used to parse the raw data from phonopy
 */
class PhonopyParser {
public:
  /** parsing of force constants.
   * @param context: the object containing the user input.
   * @return Crystal: crystal is the object describing the crystal structure.
   * @return PhononH0: the object containing the force Constants and the
   * functionality to compute the phonon energies.
   */
  static std::tuple<Crystal, PhononH0> parsePhHarmonic(Context &context);
};

// parsing helper functions for phonopy
void parseBornEffectiveCharges(Context& context, Eigen::Matrix3d& dielectricMatrix, 
                               Eigen::Tensor<double, 3>& bornCharges, 
                               std::vector<std::string>& speciesNames); 

Crystal parsePhonopyYaml(Context& context, Eigen::Vector3i& qCoarseGrid, 
                                           Eigen::MatrixXd& cellPositions2); 


#endif
