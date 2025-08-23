#ifndef PHONON_TRANSPORT_APP_H
#define PHONON_TRANSPORT_APP_H

#include "app.h"
#include "vector_bte.h"
#include "phonon_h0.h"
#include <string>

/** Main driver for the transport calculation
 */
class PhononTransportApp : public App {
public:
  void run(Context &context) override;
  void checkRequirements(Context &context) override;
 private:
  VectorBTE getPhononElectronLinewidth(Context& context, Crystal& crystalPh,
                                       ActiveBandStructure& phBandStructure,
                                       StatisticsSweep& statisticsSweep,
                                       PhononH0& phononH0);
};

#endif
