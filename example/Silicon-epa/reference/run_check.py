#!/usr/bin/env python3

import glob
import os
import json
import sys
import numpy

if __name__ == "__main__":

    listOfJsons = glob.glob("*.json")

    for filename in listOfJsons:

        filename2 = os.path.join("reference", filename)
        tol = 1e-5

        with open(filename) as f1:
            data1 = json.load(f1)
        with open(filename2) as f2:
            data2 = json.load(f2)

        print(filename)
        print(filename2)
        print(" ")

        if "onsager_coefficients" in filename:
            k1 = numpy.array(data1['electricalConductivity'])
            k2 = numpy.array(data2['electricalConductivity'])
            diff = ((k1 - k2)/max(k1)).sum()
            if abs(diff) > tol:
                print(diff, k1, k2, sep="\n")
                print(filename)
                sys.exit(1)
            k1 = numpy.array(data1['electronicThermalConductivity'])
            k2 = numpy.array(data2['electronicThermalConductivity'])
            diff = ((k1 - k2)/max(k1)).sum()
            if abs(diff) > tol:
                print(diff, k1, k2, sep="\n")
                print(filename)
                sys.exit(1)
            k1 = numpy.array(data1['seebeckCoefficient'])
            k2 = numpy.array(data2['seebeckCoefficient'])
            diff = ((k1 - k2)/max(k1)).sum()
            if abs(diff) > tol:
                print(diff, k1, k2, sep="\n")
                print(filename)
                sys.exit(1)

        if "_bands." in filename:
            k1 = numpy.array(data1['energies'])
            k2 = numpy.array(data2['energies'])
            diff = ((k1 - k2)/numpy.max(k1)).sum()
            if abs(diff) > tol:
                print(diff)
                print(filename)
                sys.exit(1)

        if "_dos." in filename:
            k1 = numpy.array(data1['dos'])
            k2 = numpy.array(data2['dos'])
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
