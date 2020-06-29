#ifndef MPICONTROLLER_H
#define MPICONTROLLER_H

#include <vector>
#include <complex>
#include <chrono>

#ifdef MPI_AVAIL 
#include <mpi.h>
#endif

class MPIcontroller{
private:
	//MPI_Comm com; // MPI communicator -- might not need this, can just call default comm "MPI_COMM_WORLD"
	int size = 0; // number of MPI processses
	int rank;

	#ifdef MPI_AVAIL 
	double startTime; // the time for the entire mpi operation
	#else 
	std::chrono::steady_clock::time_point startTime; 
	#endif
  
  int blasRank_;
  int blacsContext_;
  int numBlasRows_, numBlasCols_;
  int myBlasRow_, myBlasCol_;
  char blacsLayout_ = 'R';  // block cyclic, row major processor mapping

	public:
		// MPIcontroller class constructors -----------------------------------
		/** a constructor which sets up the MPI environment, initializes the communicator, and starts a timer **/
		MPIcontroller();
		~MPIcontroller(); 
		
		/** Calls finalize and potentially reports statistics */
		void finalize() const; 
	
		// Collective communications functions -----------------------------------
                /** Wrapper for the MPI_Broadcast function. 
                *  @param dataIn: pointer to data structure to broadcast   
                */ 
		template<typename T> void bcast(T* dataIn) const;

        /** Wrapper for MPI_Reduce in the case of a summation.
        * @param dataIn: pointer to sent data from each rank.
        * @param dataOut: pointer to buffer to receive summed data.
        */
        template<typename T> void allReduceSum(T* dataIn, T* dataOut) const;

        /** Wrapper for MPI_AllReduce in the case of a summation in-place.
         * @param data: pointer to sent data from each rank.
         * Gets overwritten with the result of the MPI allreduce('SUM') operation.
         */
        template <typename T> void allReduceSum(T* data) const;

                /** Wrapper for MPI_Reduce in the case of a summation.
                * @param dataIn: pointer to sent data from each rank,
                *       also acts as a receive buffer, as reduce is implemented IP.
                */
		template<typename T> void reduceSum(T* dataIn) const;
                /** Wrapper for MPI_Reduce which identifies the maximum of distributed data
                * @param dataIn: pointer to sent data from each rank.
                *       also acts as a receive buffer, as reduce is implemented IP.
                */
		template<typename T> void reduceMax(T* dataIn) const;
                /** Wrapper for MPI_Reduce which identifies the minimum of distributed data
                * @param dataIn: pointer to sent data from each rank.
                * @param dataOut: pointer to buffer to receive min item from data.
                */
		template<typename T> void reduceMin(T* dataIn) const;
                /** Wrapper for MPI_Gatherv which collects data from different ranks
                * and combines it into one buffer.
                * @param dataIn: pointer to sent data from each rank, with length
                *       of the number of points belonging to this rank.
                * @param dataOut: pointer to output buffer, allocated only by the
                *       head rank, of length to contain all data from all processes.
                */
                template<typename T> void gatherv(T* dataIn, T* dataOut) const;

		// point to point functions -----------------------------------
		//template<typename T> void send(T&& data) const;
		//template<typename T> void recv(T&& data) const;
	
		// Asynchronous functions
                /** Wrapper for MPI_Barrier() 
                */
		void barrier() const;
	
		// Utility functions -----------------------------------
                /** Simple function to tell us if this process is the head
                * @return isRank: returns true if this rank is the head. 
                */
		bool mpiHead() const{ return rank==0; }
                /** Function to return the rank of a process. 
                * @return rank: the rank of this process.
                */
		int getRank() const { return rank; }
                /** Function to return the number of ranks available. 
                * @return size: number of ranks 
                */
		int getSize() const { return size; }    

		// Error reporting and statistics
		void errorReport(int errCode) const; // collect errors from processes and reports them, then kills the code
		void time() const; // returns time elapsed since mpi started
	
		// IO functions
		// TODO: implement these functions, if we need them. 
		void mpiWrite();
		void mpiRead();
		void mpiAppend(); 

        // Labor division helper functions
        int workHead(); // get the first task assigned to a rank
        int workTail(); // get the last task assigned to a rank

        int getNumBlasRows();
        int getNumBlasCols();
        int getMyBlasRow();
        int getMyBlasCol();
        int getBlacsContext();

        /** Divides a number of tasks appropriately for the current MPI env.
        * @return divs: returns a vector of length 2, containing start and stop
        *       points for the divided number of tasks.
        */
        std::vector<int> divideWork(size_t numTasks); // divide up a set of work
        std::vector<int> divideWorkIter(size_t numTasks);
private:
        // store labor division information
        std::vector<int> workDivisionHeads; // start points for each rank's work
        std::vector<int> workDivisionTails; // end points for each rank's work
};

// we need to use the concept of a "type traits" object to serialize the standard cpp types
// and then we can define specific implementations of this for types which are not standard
namespace mpiContainer {
        #ifdef MPI_AVAIL
        // Forward declaration for a basic container type 
        template <typename ...> struct containerType;

        // Define a macro to shorthand define this container for scalar types
        // Size for basic scalar types will always be 1
        #define MPIDataType(cppType,mpiType) \
        template<> struct containerType<cppType> { \
                        static inline cppType* getAddress(cppType* data) { return data; } \
                        static inline size_t getSize(cppType* data) { if(!data) return 0; return 1; } \
                        static inline MPI_Datatype getMPItype() { return mpiType; } \
        };

        // Use definition to generate containers for scalar types
        MPIDataType(int, MPI_INT)
        MPIDataType(long, MPI_LONG)
        MPIDataType(unsigned int, MPI_UNSIGNED)
        MPIDataType(float, MPI_FLOAT)
        MPIDataType(double, MPI_DOUBLE)

        MPIDataType(std::complex<double>, MPI_DOUBLE_COMPLEX)
        MPIDataType(std::complex<float>, MPI_COMPLEX)
        #undef MPIDataType 

        // A container for a std::vector
        template<typename T> struct containerType<std::vector<T>>{
                        static inline T* getAddress(std::vector<T>* data) { return data->data(); }
                        static inline size_t getSize(std::vector<T>* data) { return data->size(); }
                        static inline MPI_Datatype getMPItype() { return containerType<T>::getMPItype(); }
        };
        // TODO: Define any other useful containers (a matrix? a standard array?)

        #endif
}

// Collective communications functions -----------------------------------
template<typename T> void MPIcontroller::bcast(T* dataIn) const{
        using namespace mpiContainer;
        #ifdef MPI_AVAIL
        if(size==1) return;
        int errCode;

        errCode = MPI_Bcast( containerType<T>::getAddress(dataIn), containerType<T>::getSize(dataIn), containerType<T>::getMPItype(), 0, MPI_COMM_WORLD);
        if(errCode != MPI_SUCCESS) {  errorReport(errCode); }
        #endif
}

template<typename T> void MPIcontroller::reduceSum(T* dataIn) const{
        using namespace mpiContainer;
        #ifdef MPI_AVAIL
        if(size==1) return;
        int errCode;

        if(rank==0){
                errCode = MPI_Reduce(MPI_IN_PLACE, containerType<T>::getAddress(dataIn),
                        containerType<T>::getSize(dataIn),containerType<T>::getMPItype(), MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {  
                errCode = MPI_Reduce(containerType<T>::getAddress(dataIn),containerType<T>::getAddress(dataIn),
                        containerType<T>::getSize(dataIn),containerType<T>::getMPItype(), MPI_SUM, 0, MPI_COMM_WORLD);
        } 
        if(errCode != MPI_SUCCESS) {  errorReport(errCode); }
        #endif
}

template <typename T>
void MPIcontroller::allReduceSum(T* dataIn, T* dataOut) const {
  using namespace mpiContainer;
#ifdef MPI_AVAIL
  if (size == 1) return;
  int errCode;

  errCode = MPI_Allreduce(
      containerType<T>::getAddress(dataIn),
      containerType<T>::getAddress(dataOut), containerType<T>::getSize(dataIn),
      containerType<T>::getMPItype(), MPI_SUM, MPI_COMM_WORLD);
  if (errCode != MPI_SUCCESS) {
    errorReport(errCode);
  }
#endif
}

//template <typename T>
//void MPIcontroller::allReduceSum(T* data) const {
//  using namespace mpiContainer;
//#ifdef MPI_AVAIL
//  if (size == 1) return;
//  int errCode;
//  errCode =
//      MPI_Allreduce(MPI_IN_PLACE, containerType<T>::getAddress(data),
//                    containerType<T>::getSize(data),
//                    containerType<T>::getMPItype(), MPI_SUM, MPI_COMM_WORLD);
//  if (errCode != MPI_SUCCESS) {
//    errorReport(errCode);
//  }
//#endif
//}

template<typename T> void MPIcontroller::reduceMax(T* dataIn) const{
        using namespace mpiContainer;
        #ifdef MPI_AVAIL
        if(size==1) return;
        int errCode;

        if(rank==0){
                errCode = MPI_Reduce(MPI_IN_PLACE, containerType<T>::getAddress(dataIn),containerType<T>::getSize(dataIn),
                        containerType<T>::getMPItype(), MPI_MAX, 0, MPI_COMM_WORLD);
        }
        else {
                errCode = MPI_Reduce(containerType<T>::getAddress(dataIn),containerType<T>::getAddress(dataIn),
                        containerType<T>::getSize(dataIn),containerType<T>::getMPItype(), MPI_MAX, 0, MPI_COMM_WORLD);
        }
        if(errCode != MPI_SUCCESS) {  errorReport(errCode); }
        #endif
}

template<typename T> void MPIcontroller::reduceMin(T* dataIn) const{
        using namespace mpiContainer;
        #ifdef MPI_AVAIL
        if(size==1) return;
        int errCode;

        if(rank==0){
                errCode = MPI_Reduce(MPI_IN_PLACE, containerType<T>::getAddress(dataIn),
                        containerType<T>::getSize(dataIn),containerType<T>::getMPItype(), MPI_MIN, 0, MPI_COMM_WORLD);
        }
        else {
                errCode = MPI_Reduce(containerType<T>::getAddress(dataIn),containerType<T>::getAddress(dataIn),
                        containerType<T>::getSize(dataIn),containerType<T>::getMPItype(), MPI_MIN, 0, MPI_COMM_WORLD);
        }
        if(errCode != MPI_SUCCESS) {  errorReport(errCode); }
        #endif
}

template<typename T> void MPIcontroller::gatherv(T* dataIn, T* dataOut) const {
        using namespace mpiContainer; 
        #ifdef MPI_AVAIL
        int errCode; 
        int numTasks = containerType<T>::getSize(dataOut); 
        std::vector<int> workDivs(size); 
        std::vector<int> workDivisionHeads(size); // start points for each rank's work
        // Recreate work division instructions
        for(int i = 0; i<size; i++) { 
                workDivs[i] =  (numTasks * (rank+1))/size - (numTasks * rank)/size; 
                workDivisionHeads[i] = (numTasks * i)/size;
        } 
        errCode = MPI_Gatherv(containerType<T>::getAddress(dataIn),containerType<T>::getSize(dataIn),
                containerType<T>::getMPItype(),containerType<T>::getAddress(dataOut), workDivs.data(), 
                workDivisionHeads.data(), containerType<T>::getMPItype(), 0, MPI_COMM_WORLD); 
        if(errCode != MPI_SUCCESS) {  errorReport(errCode); }
        #else
        dataOut = dataIn; // just switch the pointers in serial case 
        #endif
}

#endif
