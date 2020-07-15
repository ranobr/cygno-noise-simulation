# cygno-noise-simulation

Algorithm to generate the simulated noise images

# How-to-Run
This code is splited in two steps:
- ECDF generation;
- Images generation.

## ECDF generation:

`python2 NoiseSimCreateEcdf.py -n 100 -y 16 -r histograms_Run0000X.root`

- *n* is number of images to be taken into account to create the ECDF.
- *y* is the number of lines to be used in each chunk (in order to prevent out of memory).
- *r* is the .root filename to be used.

p.s. the output of this algorithm is a *.npy* file with the ECDF for each pixel.

## Images generation:

`python2 NoiseSimCreateImgs.py -n 100 -r 90000 -f ecdf_map_Run02054.npy`

- *n* is number of images to be generated.
- *r* is the output run number to be used as a name for the output file.
- *f* is the .npy ECDF filename to be used.


# Dependences
- Python 2.X
- Root 6.X
- root-numpy
- Numpy
- Time

