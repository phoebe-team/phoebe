@mainpage Introduction

This section of the manual covers more technical details useful to a developer.


# Apps #

The main program does very few things: intializes the paralllel environment, loads a subprogram (App) using an AppFactory, and launches it. Currently, supported apps are:
* PhononTransportApp
* ElectronWannierTransportApp
* TransportEpa
* PhononDosApp
* ElectronWannierDosApp
* ElectronFourierDosApp
* PhononBandsApp
* ElectronWannierBandsApp
* ElectronFourierBandsApp

@section scheme Code schematics

To have a good overview of the main Phoebe classes and how these work, it's easier to start with a graphical sketch of the PhononTransportApp and its code workflow.

![Fig.1: appScheme] (../images/TransportCode.png)

We refer to the class documentation for a detailed overview of all the classes.





@section Format Code formatting

Use the camelCase notation for variable definitions.
Classes should start with a capital letter.
The code should be formatted according to the clang-format style.
Classes and files should be documented using a Doxygen-compatible style, in order to render nicely in this documentation.

Name variables with sensible and clear names or the code can easily become a mess.

Use two spaces for indentation.
We are typically using CLion to format the source text, using its default code style.








@section Debugging
Valgrind is a great programming tool for memory debugging, memory leak detection and profiling.
To launch it with phoebe, an example of syntax is:
~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c}
valgrind --track-origins=yes --leak-check=full ./build/phoebe -in phoebe.in > phoebe.out &> valgrind.out
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This command launches the default valgrind tool, which is, "memcheck", which helps finding bugs in the code.

**Tip for running valgrind when the code is compiled with OpenMPI**:
Valgrind detects some false positive memory leaks, that originate from the MPI library and that you shouldn't worry about.
OpenMPI provides a file "openmpi-valgrind.supp", that can be used to suppress these wrong memory leaks messages.
The syntax is
~~~~~~~~~~~~~~~~~~~~~~~~~~~{.c }
valgrind --track-origins=yes --leak-check=full --suppressions=/usr/share/openmpi/openmpi-valgrind.supp ./build/phoebe -in phoebe.in > phoebe.out &> valgrind.out
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The "suppressions" flag is only relevant if you are compiling the code with MPI, and the path to the file "openmpi-valgrind.supp" should be modified to match the path on your computer.
