#!/usr/bin/env python
#
# R. A. Nobrega and I. Abritta July 2020
# Tool to simulated noise images
#
import numpy as np
import time
import glob
import os
import ROOT
from root_numpy import hist2array

def merge(runnumber):
    t0    = time.time()
    print("MERGING AND REMOVING CHUNK FILES")
    arrays = []
    k = 0
    string = "ecdf_map_Run"+runnumber+"chunk*.npy"
    name = glob.glob(string)
    name.sort()
    for name in name:
        print(name)
        arrays.append(np.load(name))
        os.remove(name) 
        k = k+1
 
    print(">> Saving file...")
    string = "ecdf_map_Run"+runnumber
    np.save(string, np.concatenate(arrays))
    print(">> File saved as %s" % (string))
    print("Merging elapsed time: %.2f" % (time.time()-t0))
         
    

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

## Genereta ECDF
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
        return

    t0    = time.time()
    if options.nimages:
        N     = options.nimages   # Number of images to be analyzed (usual max number of images)
    else:
        N = len(pic)
    iTr = 0
    Njump = 5 #ignore first images (usually they are unstable)
    Nx    = np.shape(hist2array(f.Get(pic[iTr])))[0] # number of pixes in a column
    Ny    = int(options.ny) # Ny represeting the number of rows at time

    ## Matrix variable to receive N images
    img_real = np.zeros((N-Njump, Ny, Nx), dtype=np.uint16)   

    for i in range(0,int(Nx/Ny)): #(Nx/Ny)
        print("PROCESSING PART %d of %d" % (i,int(Nx/Ny)))
        print(">> Loading images...")
        k = -1
        for iTr in range(Njump,N):
            k = k+1
            if k == 500 or k == 0:
                f.Close()
                f = ROOT.TFile(filename)
                pic, wfm = root_TH2_name(f)
                del wfm
            img_real[k,:,:] = hist2array(f.Get(pic[iTr]))[(Ny*i):(Ny*(i+1)),:]

        print("Time elapsed after loading: %.2f" % (time.time()-t0))
        # ECDF
        print(">> ECDF construction...")
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
            print(">> Saving chunk file...")
            string = outputfilename+'chunk'+str(i).zfill(2)
            np.save(string, np.array(ecdf_map))

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
        merge(options.filename.split('Run')[1].split('.')[0])
