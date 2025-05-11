#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def punchPlotTau(plotFileName, tau, points, pathTicks, pathLabels):
    nbands = len(tau[0,:])

    # plot the lifetimes, colored by band, for all dimensions
    plt.figure(figsize=(6,4.2))
    colors = plt.get_cmap('winter')(np.linspace(0,1,nbands))
    for ib in range(nbands):
        y = tau[:,ib]
        x = points

        indexes = np.where(y==0.)
        for index in indexes:
            try:
                y[index] = y[index-1]
            except IndexError:
                try:
                    y[index] = y[index+1]
                except IndexError:
                    raise IndexError("Not enough points in path?")

        plt.plot(x, y, label="band #{}".format(ib+1))

    # plot aesthetics
    plt.yscale('log')
    plt.ylabel(r'$\tau_{' + data['particleType'] + '}$ [' +
               data['relaxationTimeUnit'] + ']',fontsize=12)
    plt.xlim(points[0],points[-1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Find limits of the y axis
    flattenedTau = tau.flatten()
    zeroIndex = np.argwhere(flattenedTau==0.)
    flattenedTau = np.delete(flattenedTau, zeroIndex)
    ymin = 10**np.floor(np.log10(np.min(flattenedTau)))
    ymax = 10**np.ceil(np.log10(np.max(flattenedTau)))
    plt.ylim(ymin, ymax)

    plt.xticks(pathTicks,pathLabels,fontsize=12)
    for i in pathTicks:
        plt.axvline(i, color='grey')

    plt.savefig(plotFileName,bbox_inches='tight')
    plt.show(block=False)

#--------------------------------

def punchPlotBandTau(plotFileName2, energy, linewidth,
                     points, pathTicks, pathLabels, mu=None):

    if data['particleType']=="phonon":
        magFactor=10
    else:
        magFactor = 5.

    # plot some vertical lines at high sym points
    plt.figure(figsize=(5.5,5))
    for i in pathTicks:
        plt.axvline(i, color='grey')

    # if mu was calculated for electrons, shift by mu and plot line
    energyLabel = ''
    if mu is not None:
        energyLabel += r'E-E$_F$'
        plt.axhline(0., color='grey', ls='--')
        energy -= mu
    else:
        energyLabel += 'Energy'

    if magFactor == 1.:
        energyLabel += r' $\pm$ linewith'
    else:
        energyLabel += r' $\pm$ {}$\cdot$linewith'.format(magFactor)

    energyLabel += ' [' + data2['energyUnit'] +']'

    # plot the bands
    numBands = len(energies[0,:])
    for i in range(numBands):
        plt.plot(points, energies[:,i], 'k-', color='royalblue')

        error = linewidth[:,i]
        plt.fill_between(points,
                         energies[:,i] - error*magFactor,
                         energies[:,i] + error*magFactor,
                         color="#62e4e5",alpha=0.5)

    # plot aesthetics
    plt.xticks(pathTicks,pathLabels,fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(energyLabel,fontsize=14)
    plt.ylim(None, None)
    plt.xlim(points[0],points[-1])

    plt.axhline(0, color='grey', ls='-')

    plt.savefig(plotFileName2,bbox_inches='tight')
    plt.show(block=False)

#--------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot relaxation times on a band path, using the format: \n"
                                     + "python tau_path.py rta_el_relaxation_times.json path_el_bandstructure.json 0")
    parser.add_argument("INPUT",
                        help="Name of the JSON file with relaxation times")
    parser.add_argument("INPUT2",
                        help="Name of the JSON file with bandstructure on path")
    parser.add_argument("calcIndex",
                        help="Number representing index of temperature/doping calc #",
                        default=0)
    args = parser.parse_args()

    # load in the json output
    jfileName = args.INPUT
    with open(jfileName) as jfile:
        data = json.load(jfile)

        # some relaxation times may be None (e.g. acoustic phonon modes at Gamma)
        # we replace that with 0, in order to be plotted
        try:
            data['relaxationTimes'] =[ [ [ 0. if x==None else x for x in y]
                                         for y in z ]
                                       for z in data['relaxationTimes'] ]
        except KeyError:
            raise KeyError("relaxation times not found."
                           "Are you using the correct input json file?")

    # unpack the json file
    tau = np.array(data['relaxationTimes'])    # dimensions (iCalc, ik, ib)
    lwidths = np.array(data['linewidths'])    # dimensions (iCalc, ik, ib)
    mu = np.array(data['chemicalPotentials'])
    T = np.array(data['temperatures'])

    # the index used to select the calculation
    # also corresponds to the index for the temperature
    # and chemical potential of that calculation as stored in those arrays.
    calcIndex = int(args.calcIndex)
    tau = tau[calcIndex]
    lwidths = lwidths[calcIndex]
    mu = mu[calcIndex]
    print("Calculation Temperature: ", T[calcIndex], "Calculation Chemical Potential:", mu)

    # Load the bandstructure file with the kpoints
    jfileName2 = args.INPUT2
    with open(jfileName2) as jfile:
        data2 = json.load(jfile)
    # unpack the json file
    try:
        pathLabels = data2['highSymLabels']
    except KeyError:
        raise KeyError("highSymLabels not found. "
                       "Are you using the correct input json file?")
    pathTicks = data2['highSymIndices']
    points = np.array(data2['wavevectorIndices'])
    energies = np.array(data2['energies'])

    plotFileName = os.path.splitext(jfileName)[0] + ".tau.pdf"
    punchPlotTau(plotFileName, tau, points, pathTicks, pathLabels)

    plotFileName2 = os.path.splitext(jfileName2)[0] + ".tau.pdf"
    punchPlotBandTau(plotFileName2, energies, lwidths,
                     points, pathTicks, pathLabels, mu)
