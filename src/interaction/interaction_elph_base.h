#ifndef INTERACTION_ELPH_BASE_H
#define INTERACTION_ELPH_BASE_H

#include <complex>
#include <vector>
#include <stdexcept>
#include <memory>
#include "constants.h"
#include "crystal.h"
#include "eigen.h"
#include "phonon_h0.h"
#include "points.h"
#include "utilities.h"
#include "context.h"
#include "common_kokkos.h"
#include <Kokkos_Core.hpp>


class InteractionElPhWan;
class InteractionElPhSVD;

/** Abstract base class for electron-phonon interaction calculations.
 * This class provides a common interface for different implementations of
 * electron-phonon coupling calculations.
 *
 * It holds common data and defines the core virtual methods for calculations.
 * Polar correction methods are included as concrete methods.
 */
class InteractionElPhBase {
protected:
    Crystal& crystal;
    PhononH0* phononH0 = nullptr;
    bool usePolarCorrection = false;
    int numPhBands{};
    int numElBands{};
    int numElBravaisVectors{};
    int numPhBravaisVectors{};

    // Bravais lattice vectors and degeneracies, potentially stored as Views
    // in derived classes for device access, but their sizes/counts might be
    // stored here or passed during initialization.
    // Declared here as base members if needed by base methods or for size info.
    // Derived classes are responsible for initializing these members,
    // potentially from constructor parameters.
    DoubleView2D phBravaisVectors_k;
    DoubleView1D phBravaisVectorsDegeneracies_k;
    DoubleView2D elBravaisVectors_k;
    DoubleView1D elBravaisVectorsDegeneracies_k;

    // cacheCoupling should be accessible to derived classes to store results
    // and returned by getCouplingSquared.
    // Keeping it protected as per user's provided header for now.
    std::vector<Eigen::Tensor<double, 3>> cacheCoupling;

    // phase convention options -- REFACTOR switch to ENUM
    // Giustino uses Re, Rp for R vectors
    // JDFTx uses Re, Re' R vectors 
    enum phaseConventionType {
        GiustinoPhaseConvention,
        JdftxPhaseConvention,
    }; 
    enum phaseConventionType phaseConvention;


public:
    /** Base constructor
     * This constructor handles initialization of common members like crystal, phononH0,
     * and determines if polar correction is applicable based on the provided PhononH0.
     * Derived classes will typically call this base constructor.
     * @param crystal_: reference to the crystal unit cell object.
     * @param phononH0_: pointer to the phonon dynamical matrix object (can be nullptr).
     */
    InteractionElPhBase(Crystal &crystal_, PhononH0 *phononH0_ = nullptr);


    /** Multi-parameter constructor
         * This constructor is likely called by derived classes after parsing data
         * or when constructing directly with all necessary parameters.
         * It initializes common members and dimension information.
         * @param crystal_: reference to the crystal unit cell object.
         * @param phononH0_: pointer to the phonon dynamical matrix object.
         * @param numPhBands_: number of phonon bands.
         * @param numElBands_: number of electron bands.
         * @param numElBravaisVectors_: number of electronic real-space vectors.
         * @param numPhBravaisVectors_: number of phonon real-space vectors.
         * @param phBravaisVectors_k_: phonon real-space vectors (Kokkos View).
         * @param phBravaisVectorsDegeneracies_k_: phonon real-space degeneracies (Kokkos View).
         * @param elBravaisVectors_k_: electronic real-space vectors (Kokkos View).
         * @param elBravaisVectorsDegeneracies_k_: electronic real-space degeneracies (Kokkos View).
         */
        InteractionElPhBase(Crystal &crystal_, PhononH0 *phononH0_,
                        int numPhBands_, int numElBands_,
                        int numElBravaisVectors_, int numPhBravaisVectors_,
                        const DoubleView2D &phBravaisVectors_k_,
                        const DoubleView1D &phBravaisVectorsDegeneracies_k_,
                        const DoubleView2D &elBravaisVectors_k_,
                        const DoubleView1D &elBravaisVectorsDegeneracies_k_);

    /** Virtual destructor
     * Essential for proper cleanup when deleting derived objects via base pointers.
     */
    virtual ~InteractionElPhBase() = default;


protected:
    /** Helper function to ensure phononH0 is not null before accessing its methods.
     * Throws a std::runtime_error if phononH0 is nullptr.
     */
    void requirePhononH0() const;


public:
    // Pure virtual methods - Core computational functions to be implemented by derived classes
    // These methods handle the actual calculation and caching, which varies between implementations

    /** Computes the values of the el-ph coupling strength for transitions of
     * type k1,q3 -> k2, where k1 is one fixed wavevector, and k2,q3 are
     * wavevectors running in lists of wavevectors.
     * Implementation details vary between standard and SVD approaches.
     * @param eigvec1: electron eigenvector matrix U_{mb}(k1).
     * @param eigvecs2: vector of electron eigenvectors matrix U_{mb}(k2) for a list of k2 wavevectors.
     * @param eigvecs3: vector of phonon eigenvectors, in matrix form, for the corresponding q3 wavevectors.
     * @param q3Cs: list of phonon wavevectors (cartesian coordinates).
     * @param polarData: precomputed q-dependent polar correction data for each q3C.
     */
    virtual void calcCouplingSquared(
        const Eigen::MatrixXcd &eigvec1,
        const std::vector<Eigen::MatrixXcd> &eigvecs2,
        const std::vector<Eigen::MatrixXcd> &eigvecs3,
        const std::vector<Eigen::Vector3d> &q3Cs, const Eigen::Vector3d &k1C,
        const std::vector<Eigen::VectorXcd> &polarData) = 0;

    /** Computes a partial Fourier transform over the k1/R_el variables and caches the result.
     * This precomputation is done once per k1 point.
     * @param k1C: values of the k1 cartesian coordinates.
     * @param eigvec1: Wannier rotation matrix U at point k1.
     */
    virtual void cacheElPh(const Eigen::MatrixXcd &eigvec1, const Eigen::Vector3d &k1C) = 0;

    /** Resets internal caching state, should be called when k1 changes
     * or between batches if caching is batch-dependent.
     */
    virtual void resetK1() = 0;


    /** Get the coupling for the values of the wavevectors triplet (k1,k2,q3),
    * where k1 is the wavevector used at cacheElPh(),
    * k2 (at index ik2) is the wavevector of the scattered electron in the
    * final state, and q3 = k2 - k1 is the phonon wavevector.
    * Note: this method only works AFTER calcCouplingSquared has been called for the relevant k2 point.
    * @param ik2: index of the 2nd wavevector, aligned with the list of wavevectors passed to calcCouplingSquared().
    * @return g2: a constant reference to a tensor of shape (nb1,nb2,numPhBands) with the
    * values of the coupling squared |g(ik1,ik2,iq3)|^2 for the el-ph transition k1,q3 -> k2.
    */
    virtual const Eigen::Tensor<double, 3>& getCouplingSquared(const int &ik2) const = 0;


    /** Auxiliary function to return the shape of the electron-phonon tensor
    * as stored/represented by this interaction class implementation.
    * @return (numWannier,numWannier,numPhModes,numElVectors,numPhVectors)
    * or other relevant dimensions for derived classes (e.g., SVD rank).
    */
    virtual const Eigen::VectorXi getCouplingDimensions() const = 0;

    /** Estimate the memory in bytes, occupied by the implementation-specific
     * data structures containing the coupling tensor to be interpolated or its components.
     * This is often used for memory management and batching.
     * @return a memory estimate in bytes.
     */
    virtual double getDeviceMemoryUsage() const = 0;

    /** Estimate the number of batches that the list of k2 wavevectors must be
    * split into, in order to fit in memory or optimize computation.
    * @param nk2: total number of k2 wavevectors to be split in batches.
    * @param nb1: number of bands at the k1 wavevector.
    */
    virtual int estimateNumBatches(const int &nk2, const int &nb1) const = 0;


    // Polar correction methods (implemented in base class)
    // These methods calculate the long-range Frohlich interaction component in Bloch space.
    // They are concrete as they implement well-defined physics.
    // Note: Signatures align with the original InteractionElPhWan header provided by the user.

    /** Add polar correction to the electron-phonon coupling (Bloch space).
     * This calculates the long-range (Frohlich) component of the el-ph interaction.
     * @param q3: phonon wavevector, in cartesian coordinates
     * @param ev1: electron eigenvector matrix U at k
     * @param ev2: electron eigenvector matrix U at k'
     * @param ev3: phonon eigenvector at q = k'-k
     * @return g^L: a tensor of shape (nb1,nb2,numPhBands) with the values of the long-range interaction.
     */
    Eigen::Tensor<std::complex<double>, 3> getPolarCorrection(
        const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
        const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3);


    /** Calculates the q-dependent part of the polar correction V_L before transformation to Wannier gauge (part 1).
     * This method calls the static version internally after gathering required member variables.
     * @param q3: phonon wavevector.
     * @param ev3: phonon eigenvector at q3.
     * @return x: the q-dependent part of the polar correction V_L.
     */
    Eigen::VectorXcd polarCorrectionPart1(
        const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev3);


    /** Adds the transformation of V_L to the Wannier gauge (part 2).
     * This calculates the overlap <psi|e^{i(G+q).r}|psi> part of polar correction.
     * @param ev1: electron eigenvector matrix U at k.
     * @param ev2: electron eigenvector matrix U at k'.
     * @param x: q-dependent part from polarCorrectionPart1.
     * @return tensor representing the overlap part of the polar correction.
     */
    Eigen::Tensor<std::complex<double>, 3> polarCorrectionPart2(
        const Eigen::MatrixXcd &ev1, const Eigen::MatrixXcd &ev2,
        const Eigen::VectorXcd &x);


    // Static polar correction methods (implemented in base class)
    // These are static helpers used by the instance methods, possibly for internal use or external tools.
    // Note: Signatures align with the original InteractionElPhWan header provided by the user (no dimensionality parameter).

    /** Static version of getPolarCorrection, for use in contexts where
     * a base class instance is not available (e.g., parsing or initialization).
     * Requires explicit passing of all necessary parameters (volume, dielectric matrix, etc.).
     * @param q3: phonon wavevector.
     * @param ev1: electron eigenvector matrix U at k.
     * @param ev2: electron eigenvector matrix U at k'.
     * @param ev3: phonon eigenvector at q = k'-k.
     * @param volume: unit cell volume.
     * @param reciprocalUnitCell: reciprocal unit cell matrix.
     * @param epsilon: dielectric matrix.
     * @param bornCharges: Born effective charges tensor.
     * @param atomicPositions: atomic positions matrix.
     * @param qCoarseMesh: dimensions of the coarse q-mesh.
     * @return g^L: a tensor of shape (nb1,nb2,numPhBands) with the values of the long-range interaction.
     */
    static Eigen::Tensor<std::complex<double>, 3> getPolarCorrectionStatic(
        const Eigen::Vector3d &q3, const Eigen::MatrixXcd &ev1,
        const Eigen::MatrixXcd &ev2, const Eigen::MatrixXcd &ev3,
        const double &volume, const Eigen::Matrix3d &reciprocalUnitCell,
        const Eigen::Matrix3d &epsilon,
        const Eigen::Tensor<double, 3> &bornCharges,
        const Eigen::MatrixXd &atomicPositions,
        const Eigen::Vector3i &qCoarseMesh);


    /** Static version of polarCorrectionPart1.
     * Requires explicit passing of all necessary parameters.
     * @param q3: phonon wavevector.
     * @param ev3: phonon eigenvector at q3.
     * @param volume: unit cell volume.
     * @param reciprocalUnitCell: reciprocal unit cell matrix.
     * @param epsilon: dielectric matrix.
     * @param bornCharges: Born effective charges tensor.
     * @param atomicPositions: atomic positions matrix.
     * @param qCoarseMesh: dimensions of the coarse q-mesh.
     * @return x: the q-dependent part of the polar correction V_L.
     */
    static Eigen::VectorXcd polarCorrectionPart1Static(
        const Eigen::Vector3d& q3, const Eigen::MatrixXcd& ev3,
        const double& volume, const Eigen::Matrix3d& reciprocalUnitCell,
        const Eigen::Matrix3d &epsilon, const Eigen::Tensor<double, 3> &bornCharges,
        const Eigen::MatrixXd &atomicPositions, const Eigen::Vector3i &qCoarseMesh);


    /** Static version of polarCorrectionPart2.
     * Requires explicit passing of all necessary parameters.
     * @param ev1: electron eigenvector matrix U at k.
     * @param ev2: electron eigenvector matrix U at k'.
     * @param x: q-dependent part from polarCorrectionPart1.
     * @return tensor representing the overlap part of the polar correction.
     */
    static Eigen::Tensor<std::complex<double>, 3> polarCorrectionPart2Static(
        const Eigen::MatrixXcd &ev1, const Eigen::MatrixXcd &ev2,
        const Eigen::VectorXcd &x);


    // Note: precomputeQDependentPolar is not included in base as per user request,
    // although it was present in the original InteractionElPhWan implementation.
};

#endif
