#ifndef UTILS_H
#define UTILS_H

#include <tuple>
#include <string>
#include <vector>
#include "eigen.h"

// returns the remainder of the division between two integers.
// Note: c++ defines the % operator such that: (a/b)*b + a%b == a    (for b!=0)
// this works as mod() in fortran or % in python when a and b are positive
// But the behavior is not the same for negative integers
// the function below can be used instead
int mod(const int &a, const int &b);

// returns -1 if val<0, 0 if val=0, and 1 if val>0
template<typename T> int sgn(T &val) {
    return (T(0) < val) - (val < T(0));
}

int compress3Indices(const int &i1, const int &i2, const int &i3,
        const int &size1, const int &size2, const int &size3);

std::tuple<int, int, int> decompress3Indices(const int &iTot,
        const int &size1, const int &size2, const int &size3);

int compress2Indices(const int &i1, const int &i2, const int &size1,
        const int &size2);

std::tuple<int, int> decompress2Indices(const int &iTot, const int &size1,
        const int &size2);

// helper to break up strings by commas and spaces
std::vector<std::string> tokenize(const std::string str);

// A function to allocate a dynamically sized array. It tricks the
// compiler into thinking the size is a constant via the const identifier
// on the argument. This resolves issues with VLAs -- see crystal.cpp
template <typename T> T* allocate(T *&array, const unsigned int size){
        array = new T [size];
        return array;
}

/** Class for implementing strong typing.
 * In fact, when using the methods (de)compress2(3)indices, it's easy to
 * mix the order of indices. This class can be used to enforce the type
 * of the index, without incurring in a penalty in the code speed.
 * Taken from https://www.fluentcpp.com/2017/05/05/news-strong-types-are-free/
 */
template<typename T, typename Parameter>
class NamedType {
public:
    explicit NamedType(T const &value) :
            value_(value) {
    }
    T& get() {
        return value_;
    }
    T const& get() const {
        return value_;
    }
private:
    T value_;
};

/** CartIndex is used to label cartesian directions
 */
using CartIndex = NamedType<int, struct CartTag>;

/** BteIndex is used to label the Bloch states entering the BTE
 */
using BteIndex = NamedType<int, struct BteTag>;

/** CalcIndex is used to label the pairs of (chemical potential , temperature)
 * used in transport calculations
 */
using TempIndex = NamedType<int, struct TempTag>;
using ChemPotIndex = NamedType<int, struct ChemPotTag>;

/** WavevectorIndex is used to label k/q points
 */
using WavevectorIndex = NamedType<int, struct WavevectorTag>;

/** BandIndex is used to label bands at a given k/q points
 */
using BandIndex = NamedType<int, struct BandTag>;

/** StateIndex is used to label the Bloch states in the band structure.
 * Not always equal to BteIndex, because some states may be discarded!
 */
using StateIndex = NamedType<int, struct StateTag>;

/** Function to obtain an estimate of virtual memory used so far.
 *
 * @return [vm_usage,resident_set]: tuple with
 * 1) memory used by the OS in MB
 * 2) memory used by phoebe in MB
 */
std::tuple<double, double> memoryUsage();

/** splitVector is a utility that splits a std::vector<> into chunks with
 * specified size.
 *
 * @param v: input vector to be split
 * @param chunkSize: (max) size of the chunk obtained after splitting
 * @return std::vector<std::vector<T>>: a vector of chunks (vectors), containing
 * the split copy of the input.
 */
template<typename Vector>
std::vector<Vector> splitVector(const Vector& v, unsigned chunkSize) {
  using Iterator = typename Vector::const_iterator;
  std::vector<Vector> vectorChunks;
  Iterator it = v.cbegin();
  const Iterator end = v.cend();

  while (it != end) {
    Vector v2;
    std::back_insert_iterator<Vector> inserter(v2);
    const auto num_to_copy = std::min(static_cast<unsigned>(
                                          std::distance(it, end)), chunkSize);
    std::copy(it, it + num_to_copy, inserter);
    vectorChunks.push_back(std::move(v2));
    std::advance(it, num_to_copy);
  }

  return vectorChunks;
}

double findMaxRelativeDifference(const Eigen::Tensor<double,3> &x,
                                 const Eigen::Tensor<double,3> &xRef);

/** Function to split a string into tokens 
 * @param s : the string to be split
 * @param delimiter: the character by which to split the string 
 */
std::vector<std::string> split(const std::string &s, char delimiter);

#endif

