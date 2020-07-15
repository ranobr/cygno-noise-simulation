#!/usr/bin/env python
#
# R. A. Nobrega and I. Abritta Febraury 2019
# Tool to simulated noise images
#
import numpy as np
import commands
import time
import ROOT
from root_numpy import hist2array, array2hist


# generate simulated data using a ECDF curve (x,y)
def clone_hist(x, y, Nclone):
    s = np.random.uniform(0,1,Nclone)
    idx = np.searchsorted(y, s, side='left')
    return x[idx]

def generateNoiseImage(ecdf_file,Nclone):
    ecdf_map_load = np.load(ecdf_file,allow_pickle = True,fix_imports=True)
    
    Nx = np.size(ecdf_map_load, 0)
    Ny = np.size(ecdf_map_load, 1)

    img_sim = np.zeros((Nclone, Nx, Ny), dtype=np.uint16)

    for i in range(0, Nx):
        for j in range(0, Ny):
            x = ecdf_map_load[i][j][0]
            y = ecdf_map_load[i][j][1]    
            img_sim[:,i,j] = clone_hist(x, y, Nclone)
    return img_sim

def generate_noise(options):
    t0 = time.time()
    
    Nclone = options.nimages
    run_count = str(options.run).zfill(5)

    infolder  = ""
    infile    = "histograms"
    outfolder = "/"

    ecdf_filename = options.filename

    outfile=ROOT.TFile(outfolder+'/'+infile+'_Run'+run_count+'.root', 'RECREATE') #OUTPUT NAME

    imgs = generateNoiseImage(ecdf_filename, Nclone)
    nX = np.shape(imgs)[1]
    nY = np.shape(imgs)[2]

    print('[Generating images]')
    for entry in range(0, Nclone): #RUNNING ON ENTRIES

        final_imgs = ROOT.TH2F('pic_run'+str(run_count)+'_ev'+str(entry), '', nX, 0, (nX-1), nY, 0, (nY-1))
        total = imgs[entry,:,:]

        array2hist(total, final_imgs)
        final_imgs.Write()

    print('[Run %d with %d images complete]'%(int(run_count),Nclone))
    #run_count+=1
    outfile.Close()


    t1=time.time()
    print('Generation took %.2f seconds'%(t1-t0))
    return
                
            
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage='NoiseSimCreateImgs.py [OPTION1,..,OPTIONN]\n\n Example of use:\n\t [ecdf]    NoiseSimCreateImgs.py -n 100 -r 90000 -f ecdf_map_Run02054.npy')
    parser.add_option('-n','--nimages', dest='nimages', type='int', default=100, help='Number of imagens to be create');
    parser.add_option('-r','--run', dest='run', type='int', default=9000, help='output runnumber');
    parser.add_option('-f','--file', dest='filename', type='string', default='', help='Filename [get] or path/filename[put] to the desired file.');
    #parser.add_option('-h', '--help', help='Show this help message');
    (options, args) = parser.parse_args()

    if (options.filename == ''):
        print('Filename/Directory is necessary to run this script')
        exit()
    else:
        generate_noise(options)
