#!/usr/bin/python
import numpy as np
import json
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 14})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot relaxation times along a path "
                                     "as a colormap along the band structure, using the format: \n"
                                     + "python tau_path.py rta_el_relaxation_times.json path_el_bandstructure.json 0")
    parser.add_argument("INPUT",
                        help="Name of the JSON file with relaxation times")
    parser.add_argument("INPUT2",
                        help="Name of the JSON file with bandstructure on path")
    parser.add_argument("calcIndex",
                        help="Number representing index of temperature/doping calc #",
                        default=0)
    args = parser.parse_args()

# open the data sets
dataTau =  json.load(open(args.INPUT))
dataPath =  json.load(open(args.INPUT2))
calcIndex = int(args.calcIndex)

# unpack the json files
tau = np.array(dataTau['relaxationTimes'])    # dimensions (iCalc, ik, ib)
linewidths = np.array(dataTau['linewidths'])      # dimensions (iCalc, ik, ib)
mu = np.array(dataTau['chemicalPotentials'])
T = np.array(dataTau['temperatures'])

particleType = dataPath['particleType']
energies = np.array(dataPath['energies'])      # dimensions (iCalc, ik, ib)
pathTicks = dataPath['highSymIndices']
pathLabels = dataPath['highSymLabels']
points = np.array(dataPath['wavevectorIndices'])
numBands = dataPath['numBands']

# select which calculation to plot
linewidths = linewidths[calcIndex]
tau = np.array(tau[calcIndex])

def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap

# set up the color map
c = ['#62e4e5','#5E4AFF','#000087' ] # bluescale
cm = custom_div_cmap(256, mincol=c[0], midcol=c[1] ,maxcol=c[2])

# set up figure
fig = plt.figure(figsize=(5.5,5))
ax = fig.add_subplot(1, 1, 1)

# flatten the band path data
allPoints = []
for i in range(numBands):
    allPoints.append(points)
allPoints = np.array(allPoints).T

# plot the data as a colored scatter plot
cf = ax.scatter(allPoints, (energies-mu), marker='o',
        cmap=cm, s=18, linewidth=0, c=linewidths, alpha=0.75, zorder=2)

# plot vertical lines at high sym points
for i in pathTicks:
    ax.axvline(i, color='grey')

# plot the chemical potential if these are electrons
if(particleType == 'electron'):
    ax.axhline(0, color='grey', ls='--')
    energyLabel = r'E-$\mu$ [' + dataPath['energyUnit'] +']'
else:
    energyLabel = 'Energy [' + dataPath['energyUnit'] +']'

# colorbar
divider = make_axes_locatable(ax)
cax = divider.new_horizontal(size="4%", pad=0.135)
fig.add_axes(cax)
cb = fig.colorbar(cf,cax=cax,orientation="vertical")
cb.set_alpha(1)
cb.draw_all()
cb.set_label(r'$\Gamma_{\mathrm{'+ particleType[:2] +'-ph}}$ [' +dataTau['linewidthsUnit']+']',
        labelpad=25,rotation=-90)
cb.outline.set_visible(False)

# axis settings
ax.set_xticks(pathTicks)
ax.set_xticklabels(pathLabels)
ax.set_ylabel(energyLabel)
ax.set_ylim(None, None)
ax.set_xlim(points[0],points[-1])

plt.savefig('color_bands.png',bbox_inches='tight', dpi=200)
