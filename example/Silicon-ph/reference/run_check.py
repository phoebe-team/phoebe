#!/usr/bin/env python3

import glob
import os
import json
import sys
import numpy

if __name__ == "__main__":

    tol = 1e-4 # test for 0.01% changes
    listOfJsons = glob.glob("*.json")

    for filename in listOfJsons:

        filename2 = os.path.join("reference", filename)

        with open(filename) as f1:
            data1 = json.load(f1)
        with open(filename2) as f2:
            data2 = json.load(f2)

        print(filename)
        print(filename2)
        print(" ")

        if "thermal_cond" in filename:
            k1 = numpy.array(data1['thermalConductivity'])
            k2 = numpy.array(data2['thermalConductivity'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
          #  print(diff, k1, k2, (k1 - k2), max(k1),sep="\n")
            #sys.exit(1)
            if abs(diff) > tol*5: # relaxons solver has a slight variation in small case
                print(diff, k1, k2, sep="\n")
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol:
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

        if "specific_heat" in filename:
            k1 = numpy.array(data1['specificHeat'])
            k2 = numpy.array(data2['specificHeat'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol: # viscosities are small
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

        if "viscosity" in filename:
            k1 = numpy.array(data1['phononViscosity'])
            k2 = numpy.array(data2['phononViscosity'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol: # viscosities are small
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol: # viscosities are small
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

        if "real_space" in filename:
            k1 = numpy.array(data1['specificHeat'])
            k2 = numpy.array(data2['specificHeat'])
            diff = (k1 - k2)/numpy.max(k1)
            if abs(diff) > tol: # viscosities are small
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol:
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

            k1 = numpy.array(data1['Ai'])
            k2 = numpy.array(data2['Ai'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol: # viscosities are small
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

            k1 = numpy.array(data1['Du'])
            k2 = numpy.array(data2['Du'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if abs(diff2) > tol:
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

            k1 = numpy.array(data1['Wji0'])
            k2 = numpy.array(data2['Wji0'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
            if diff2 > tol:
                print("failed max element check",diff2)
                print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
                print("max element difference", numpy.max(k1-k2))
                print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
                print(filename)
                sys.exit(1)

        if "path_" in filename and "_relaxation_times" in filename:
            k1 = numpy.array(data1['linewidths'])
            k2 = numpy.array(data2['linewidths'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            k1 = numpy.array(data1['energies'])
            k2 = numpy.array(data2['energies'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            k1 = numpy.array(data1['velocities'])
            k2 = numpy.array(data2['velocities'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)
            k1 = numpy.array(data1['relaxationTimes'])
            k2 = numpy.array(data2['relaxationTimes'])
            k1[numpy.where(k1 == None)] = 0
            k2[numpy.where(k2 == None)] = 0
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)



    print("Reference checks Done")
    sys.exit(0)
