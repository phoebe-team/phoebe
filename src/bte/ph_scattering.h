#ifndef PHSCATTERING_H
#define PHSCATTERING_H

#include "scattering.h"
#include "vector_bte.h"
#include "interaction_3ph.h"
#include "phonon_h0.h"

class PhScatteringMatrix : public ScatteringMatrix {
public:
	PhScatteringMatrix(Context & context_,
			PhStatisticsSweep & phStatisticsSweep_,
			FullBandStructure<FullPoints> & innerBandStructure_,
			FullBandStructure<FullPoints> & outerBandStructure_,
			Interaction3Ph * coupling3Ph_=nullptr,
			PhononH0 * h0=nullptr);
//			InteractionIsotope * couplingIsotope_=nullptr,
//			InteractionBoundary * couplingBoundary_=nullptr

	PhScatteringMatrix(const PhScatteringMatrix & that);
	PhScatteringMatrix & operator=(const PhScatteringMatrix & that);

protected:
	PhStatisticsSweep & phStatisticsSweep;

	Interaction3Ph * coupling3Ph;
//	InteractionIsotope * couplingIsotope = nullptr;
//	InteractionBoundary * couplingBoundary = nullptr;
	PhononH0 * h0;

	virtual void builder(Eigen::MatrixXd * matrix, VectorBTE * linewidth,
			VectorBTE * inPopulation, VectorBTE * outPopulation);
};

#endif
