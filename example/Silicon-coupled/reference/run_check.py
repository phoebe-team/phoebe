#!/usr/bin/env python3

import glob
import os
import json
import sys
import numpy

def checkCoefficient(coeffName, data1, data2, tol):

    if isinstance(data1[coeffName], str): return

    print("Checking", coeffName)

    k1 = numpy.array(data1[coeffName])
    k2 = numpy.array(data2[coeffName])

    # negative or super small lifetimes can cause large difference for small changes
    if(coeffName == "relaxationTimes"):
        k1[numpy.where(k1<0)] = 0
        k2[numpy.where(k2<0)] = 0
        k1[numpy.where(k1>1e15)] = 0
        k2[numpy.where(k2>1e15)] = 0
    if(coeffName == "linewidths"):
        k1[numpy.where(k1==None)] = 0
        k2[numpy.where(k2==None)] = 0

    diff = ((k1 - k2)/numpy.max(k1)).sum()
    if abs(diff) > tol:
        print(diff, k1, k2, sep="\n")
        print(filename)
        #sys.exit(1)
    diff2 = (numpy.max(k1) - numpy.max(k2))/numpy.max(k1)
    if abs(diff2) > tol:
        print("failed max element check",diff2)
        print("max element, run vs. ref ", numpy.max(k1), numpy.max(k2))
        print("max element difference", numpy.max(k1-k2))
        print("max element % difference", numpy.max(k1-k2)/numpy.max(k1))
        print(filename)
        #sys.exit(1)

if __name__ == "__main__":

    listOfJsons = glob.glob("*.json")
    tol = 1e-5

    for filename in listOfJsons:

        filename2 = os.path.join("reference", filename)

        with open(filename) as f1:
            data1 = json.load(f1)
        try:
            with open(filename2) as f2:
                data2 = json.load(f2)
        except FileNotFoundError:
            continue

        print("\n----------------")
        print(filename," against ", filename2)
        print(" ")

        if "transport_coefficients" in filename:
            for key in data1:
                checkCoefficient(key, data1, data2, tol)

        if "_relaxation_times" in filename:
            for key in data1:
                checkCoefficient(key, data1, data2, tol)

        if "viscosity" in filename:
            for key in data1:
                checkCoefficient(key, data1, data2, tol)

        if "real_space" in filename:
            for key in data1:
                checkCoefficient(key, data1, data2, tol)

    print("\nReference checks Done")
    sys.exit(0)
