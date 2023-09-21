#ifndef COUPLED_TRANSPORT_APP_H
#define COUPLED_TRANSPORT_APP_H

#include "app.h"
#include <string>

/** Main driver for the transport calculation
 */
class CoupledTransportApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;

};

#endif
