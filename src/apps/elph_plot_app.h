#ifndef EL_PH_PLOT_APP_H
#define EL_PH_PLOT_APP_H

#include <string>
#include "app.h"

/** Main driver for the calculation
 */
class ElPhCouplingPlotApp: public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

#endif
