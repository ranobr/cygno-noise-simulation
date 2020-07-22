# cygno-noise-simulation

Algorithm to generate ecdf map and simulated noise images

# How-to-Run
This code is splited in two steps:
- ECDF generation;
- Images generation.

## ECDF generation:

`python3 NoiseSimCreateEcdf.py -n 100 -y 512 -r histograms_RunXXXXX.root`

- *n* number of images to be taken into account to create the ECDF. If not defined, all run images will be used. The first five images are always skipped. 
- *y* number of lines to be used in each chunk (in order to prevent out of memory however, the lower this number, the longer the processing time). This number must be an exact fraction of the number of rows of the sensor. If the code fails when trying to merge the chunk files because of memory error, use *-y -1* to run only the part of the code which merges the chunk files in disk.
- *r* is the .root filename to be used.

p.s. the output of this algorithm is a *.npy* file containing a ECDF for each pixel. Depending on the *n* and *y* values, this routine might take 90 minutes to run (tested on the cygno.cloud.infn computer for *n*=1000 and *y*=32). To make it faster *y* value must be high.

## Images generation:

`python3 NoiseSimCreateImgs.py -n 500 -r 90000 -f ecdf_map_RunXXXXX.npy`

- *n* number of images to be generated.
- *r* output run number used in the name of the output file (e.g. histograms_Run90000.root).
- *f* is the .npy ECDF filename to be used.

p.s. the output of this algorithm is a *.root* file with the simulated images. It takes less than 1 second per image to run (tested on the cygno.cloud.infn computer producing 700 images in 500 seconds).



# Dependences
- Python 3.X
- Root 6.X
- root-numpy
- numpy
- time
- glob
- os
