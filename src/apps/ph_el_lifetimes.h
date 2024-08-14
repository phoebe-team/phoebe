#ifndef PHEL_LIFETIMES_APP_H
#define PHEL_LIFETIMES_APP_H

#include "app.h"

/** Main driver for the calculation
 */
class PhElLifetimesApp: public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
};

#endif
