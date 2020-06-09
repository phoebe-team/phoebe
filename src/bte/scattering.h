#ifndef SCATTERING_H
#define SCATTERING_H

#include "context.h"
#include "vector_bte.h"
#include "delta_function.h"

class ScatteringMatrix {
public:
	ScatteringMatrix(Context & context_, StatisticsSweep & statisticsSweep_,
			FullBandStructure<FullPoints> & innerBandStructure_,
			FullBandStructure<FullPoints> & outerBandStructure_);
	ScatteringMatrix(const ScatteringMatrix & that); // copy constructor
	ScatteringMatrix & operator=(const ScatteringMatrix & that);//assignment op
	~ScatteringMatrix();

	void setup();

	VectorBTE diagonal();
	VectorBTE offDiagonalDot(VectorBTE & inPopulation);
	VectorBTE dot(VectorBTE & inPopulation);

	VectorBTE getSingleModeTimes();
	void a2Omega();
	void omega2A();

	std::tuple<VectorBTE,Eigen::MatrixXd> diagonalize();
protected:
	Context & context;
	StatisticsSweep & statisticsSweep;
	DeltaFunction * smearing;

	FullBandStructure<FullPoints> & innerBandStructure;
	FullBandStructure<FullPoints> & outerBandStructure;

	 // constant relaxation time approximation -> the matrix is just a scalar
	bool constantRTA = false;
	bool highMemory = true;
	bool hasCGScaling = false;
	bool isMatrixOmega = false;

	VectorBTE internalDiagonal;
	Eigen::MatrixXd theMatrix;
	long numStates;
	long numPoints;
	long numCalcs;

	std::vector<long> excludeIndeces;

	// pure virtual function
	// needs an implementation in every subclass
	virtual void builder(Eigen::MatrixXd & matrix, VectorBTE * linewidth,
			VectorBTE * inPopulation, VectorBTE * outPopulation) = 0;
};

#endif
