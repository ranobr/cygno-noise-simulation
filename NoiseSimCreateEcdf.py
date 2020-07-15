#!/usr/bin/env python
#
# R. A. Nobrega and I. Abritta Febraury 2019
# Tool to simulated noise images
#
import numpy as np
import commands
import time
import ROOT
from root_numpy import hist2array
import noisesim_lib as ns


def root_TH2_name(root_file):
    pic = []
    wfm = []
    for i,e in enumerate(root_file.GetListOfKeys()):
        che = e.GetName()
        if ('pic_run' in str(che)):
            pic.append(che)
        elif ('wfm_run' in str(che)):
            wfm.append(che)
    return pic, wfm

def ecdf(data):
    """ Compute ECDF """
    unique, counts = np.unique(data, return_counts=True)
    y = np.cumsum(counts)
    y = y / float(y[-1])
    y = np.insert(y, 0, 0)
    unique = np.insert(unique, 0, unique[0]-1)
    return(unique,y)

def generate_ecdf(options):
    t1=time.time()

    filename = options.filename
    outputfilename = "ecdf_map_Run"+options.runnumber
    saveMap = 1
    try:
        f = ROOT.TFile(filename)
        pic, wfm = root_TH2_name(f)
        del wfm
    except:
        print("File not found")

    t0    = time.time()
    if options.nimages:
        N     = options.nimages   # Number of images to be analyzed (usual max number of images)
    else:
        N = len(pic)
    Njump = 5 #ignore first images (usually they are unstable)
    Nx    = int(2048) # Fixed in 2048
    Ny    = int(options.ny) # Ny represeting the number of lines at time

    ## Matrix variable to receive N images
    img_real = np.zeros((N-Njump, Ny, Nx), dtype=np.uint16)   

    for i in range(0,1): #(Nx/Ny)
        print("Doing part %d of %d" % (i,Nx/(Ny-1)))
        #img_real[k, np.array(pixel)[0,:], np.array(pixel)[1,:]] = hist2array(f.Get(pic[iTr]))[pixel]
        k = -1
        for iTr in range(Njump,N):
            k = k+1
            img_real[k,:,:] = hist2array(f.Get(pic[iTr]))[(Ny*i):(Ny*(i+1)),:]

        print("Time elapsed after loading: %.2f" % (time.time()-t0))
        # ECDF
        t0=time.time()
        ecdf_map = []
        for ii in range(0,Ny):
            ecdf_map.append([])
            for jj in range(0,Nx):    
                ecdf_map[ii].append([])
                x,y = ecdf(img_real[:,ii,jj])
                ecdf_map[ii][jj].append(x)
                ecdf_map[ii][jj].append(y)
        print("Time elapsed after ECDF: %.2f" % (time.time()-t0))
        if saveMap == 1:
            string = outputfilename+'chunk'+str(i)
            np.save(string, np.array(ecdf_map))

    # juntar as chunks
    print("Total time elapsed: %.2f" % (time.time()-t1))
    return 
                
            
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage='NoiseSimCreateEcdf.py [OPTION1,..,OPTIONN]\n\n Example of use:\n\t [ecdf]    NoiseSimCreateEcdf.py -n 100 -y 16 -r histograms_Run0000X.root')
    parser.add_option('-n','--nimages', dest='nimages', type='int', default=0, help='Number of imagens to be analyzed');
    parser.add_option('-y','--ny', dest='ny', type='int', default=4, help='Number of lines to be analyzed at time');
    parser.add_option('-r','--run', dest='filename', type='string', default='', help='Filename [get] or path/filename[put] to the desired file.');
    #parser.add_option('-h', '--help', help='Show this help message');
    (options, args) = parser.parse_args()

    if (options.filename == ''):
        print('Filename/Directory is necessary to run this script')
        exit()
    else:
        setattr(options,'runnumber', options.filename.split('Run')[1].split('.')[0])
        generate_ecdf(options)
