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
    if(coeffName == "chemicalPotentials"): return # phonon mu = 0 always
    if(coeffName == "relaxationTimes"):
        k1[numpy.where(k1==None)] = 0
        k2[numpy.where(k2==None)] = 0
    #if(coeffName == "linewidths"):
    #    k1[numpy.where(k1==None)] = 0
    #    k2[numpy.where(k2==None)] = 0

    diff = ((k1 - k2)/numpy.max(k1)).sum()
    if abs(diff) > tol:
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

if __name__ == "__main__":

    listOfJsons = glob.glob("*.json")
    tol = 1e-4

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

        if "thermal_cond" in filename:
            for key in data1:
                checkCoefficient(key, data1, data2, tol)

        if "_relaxation_times" in filename:
            temp_tol = tol
            if "relaxon" in filename: # these can vary slightly more than others
                temp_tol = 1e-3
            for key in data1:
                checkCoefficient(key, data1, data2, temp_tol)

    print("\nReference checks Done")
    sys.exit(0)
