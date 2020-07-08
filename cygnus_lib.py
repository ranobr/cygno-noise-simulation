#
# Image analisys CYGNUS-RD Python Library
# G. Mazzitelli 2017 
# rev oct 2018 - swift direct access 
#

import numpy as np
import glob, os
import re
import sys


# ############################################################################################################################ #
# DATA Archive
# ############################################################################################################################ #

def logbookInfo(file, run):
    logbook = np.loadtxt(file, dtype=str, delimiter='\t')
    for i in range(1, len(logbook[:,0])):
        if logbook[:,0][i] == str(run):
            data = logbook[i]
    return logbook[0], data

def strigHMS2time(str):
    import datetime
    pt=datetime.datetime.strptime(str,'%H:%M:%S')
    sec=pt.second+pt.minute*60+pt.hour*3600
    return sec

def ScopeHeader(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    f.close()
    size_sco      = int(str.split(lines[1])[3])-1
    start_sec_sco = strigHMS2time(str.split(lines[3])[2])
    TA_start_sco  = lines[5]
    TA_end_sco    = lines[size_sco+5]
    T_start_sco   = float(str.split(TA_start_sco)[0])
    T_end_sco     = float(str.split(TA_end_sco)[0])

    return size_sco, start_sec_sco, T_start_sco, T_end_sco

def ScopeHeaderPath(path, ch):
    sch = 'C'+str(ch)+'wave'
    if path.find('BTF_2017-2') or dataSelection == 'LABOct2017':
        sch = 'C'+str(ch)+'Run'
    file_sco = path + sch + '%05d' % (0) +'.txt'
    size_sco, start_sec_sco, T_start_sco, T_end_sco = ScopeHeader(file_sco)

    return size_sco, start_sec_sco, T_start_sco, T_end_sco

def ReadScope(path, ch):
    file_in_dir=os.listdir(path)
    sch = 'C'+str(ch)+'wave'
    if path.find('BTF_2017-2')  or dataSelection == 'LABOct2017':
        sch = 'C'+str(ch)+'Run'
    if sch in str(file_in_dir):
        nfile = np.size(filter(lambda x: sch in x, file_in_dir))
        file_sco = path + sch + '%05d' % (0) +'.txt'
        size_sco, start_sec_sco, T_start_sco, T_end_sco = ScopeHeader(file_sco)
        t = np.empty((nfile, size_sco), dtype=np.double)
        a = np.empty((nfile, size_sco), dtype=np.double)
        for i in range(0, nfile):
            file_sco = path + sch + '%05d' % (i) +'.txt'
            t[i,], a[i,] = np.loadtxt(file_sco, delimiter=' ', skiprows=6, usecols=(0, 1), 
                                      unpack=True, dtype='double')
    return t, a

def ReadScopeTrace(path):

#    t, a = np.loadtxt(path, delimiter=' ', skiprows=6, usecols=(0, 1), 
#                       unpack=True, dtype='double')
    t, a = np.genfromtxt(path, delimiter=' ', skip_header=7, unpack=(0, 1))
    return t, a

def file2PathCygnus(dataSelection, fileNumber, ftype):
    #  return ftyp file path RUN, H5, TS, LOG, SCO in dataSelection 
    itype = ['RUN', 'H5', 'TS', 'LOG', 'SCO', 'TMP']
    base_path = 'Data/'+dataSelection+'/'
    RUN_path = 'Run%03d/' % np.int(fileNumber)
    H5_path   = base_path + 'Data_Camera/H5/'
    TS_path   = base_path + 'Data_Camera/TS/'
    LOG_path  = base_path + 'LOG/'
    SCO_path  = base_path + 'Data_Scope/'
    TMP_path  = base_path + 'TMP/'
    path = [RUN_path, H5_path, TS_path, LOG_path, SCO_path, TMP_path]
    return path[itype.index(ftype)]

def file2FullPathCygnus(dataSelection, fileNumber, ftype):
    itype = ['RUN', 'H5', 'TS', 'LOG', 'SCO', 'TMP']
    est = ['.HIS', '/', '', '.LOG', '/', '']
    return file2PathCygnus(dataSelection, fileNumber, ftype)+'Run%03d' % np.int(fileNumber) + est[itype.index(ftype)]

def imageFile2FullPathCygnus(dataSelection, fileNumber, traccia):
    return file2FullPathCygnus(dataSelection, fileNumber, 'H5')+'run%03d' % np.int(fileNumber)+('-%04d.h5' % traccia)

def scopeFile2FullPathCygnus(dataSelection, fileNumber, traccia, ch):
    sch = 'C'+str(ch)+'wave'
    if dataSelection == 'BTF_2017-2' or dataSelection == 'LABOct2017':
        sch = 'C'+str(ch)+'Run'
    return file2FullPathCygnus(dataSelection, fileNumber, 'SCO') + sch + '%05d' % (traccia) +'.txt'


# ############################################################################################################################ #
# swift storage read and write
# ############################################################################################################################ #

def swift_auth():
    # https://docs.openstack.org/python-swiftclient/latest/service-api.html
    # https://docs.openstack.org/python-swiftclient/

    import swiftclient
    from keystoneauth1 import session
    from keystoneauth1.identity import v3

    
    OS_PROJECT_NAME='anonymous'
    OS_USERNAME='anonymous'
    OS_PASSWORD='anonymous'
    OS_AUTH_URL='https://keystone.cloud.infn.it:5000/v3/'
    OS_REGION_NAME='canf'
    OS_PROJECT_DOMAIN_NAME='default'
    OS_USER_DOMAIN_NAME='default'
    OS_IDENTITY_API_VERSION='3'
    OS_STORAGE_URL='https://swift.cloud.infn.it:8080/v1/AUTH_1e60fe39fba04701aa5ffc0b97871ed8'


    _auth = v3.Password(
        user_domain_name    = OS_USER_DOMAIN_NAME,
        project_domain_name = OS_PROJECT_DOMAIN_NAME,
        project_name        = OS_PROJECT_NAME,
        username            = OS_USERNAME,
        password            = OS_PASSWORD,
        auth_url            = OS_AUTH_URL
    )
    _os_options={
        'region_name' : OS_REGION_NAME, 
        'object_storage_url': OS_STORAGE_URL
    }
    # Create session
    keystone_session = session.Session(auth = _auth)

    # Create swiftclient Connection
    swift = swiftclient.Connection(session      = keystone_session, 
                                    auth_version = OS_IDENTITY_API_VERSION,
                                    os_options   = _os_options
                                    )
    return swift

def swift_auth_read_image_h5(file):
    # https://www.getdatajoy.com/learn/Read_and_Write_HDF5_from_Python#Reading_Data_from_HDF5_Files
    # dump imagine con autenticazione
    import numpy as np
    import h5py
    import os
    swift = swift_auth()
    obj_tuple = swift.get_object("Cygnus", file)
    tmpname = "./tmp." + str(os.getpid()) + ".h5"
    with open(tmpname, 'wb') as my_tmp:
            my_tmp.write(obj_tuple[1])
    image = read_image_h5(tmpname)
    try:
        os.remove(tmpname)
    except OSError:
        pass
    return image

def swift_read_image_h5(file):
    # dump imagine in sola lettura
    import requests
    import os
    BASE_URL  = "https://swift.cloud.infn.it:8080/v1/AUTH_1e60fe39fba04701aa5ffc0b97871ed8/Cygnus/"

    url = BASE_URL+file
    r = requests.get(url)
    tmpname = "./tmp." + str(os.getpid()) + ".h5"
    with open(tmpname, 'wb') as tmp:
            tmp.write(r.content)
    image = read_image_h5(tmpname)
    try:
        os.remove(tmpname)
    except OSError:
        pass
    return image

def swift_listdir(dirname):
    swift = swift_auth()
    fileindir=[]
    for data in swift.get_container("Cygnus", full_listing=True)[1]:
        if dirname in str(data):
            if data['name'].find("._") <= 0 : # Petch aggiunta per i cazzo di file mac...
                fileindir.append(data['name'])
    return fileindir

def swift_noauth_listdir(dirname):
    import requests
    BASE_URL  = "https://swift.cloud.infn.it:8080/v1/AUTH_1e60fe39fba04701aa5ffc0b97871ed8/Cygnus/"
    r = requests.get(BASE_URL)

    r = r.content
    string = r.decode("ISO-8859-1")
    dati = string.split('\n')

    fileindir=[]
    for data in dati:
        if dirname in data:
            fileindir.append(data)
    return fileindir


# ############################################################################################################################ #
# TOOLS
# ############################################################################################################################ #

def read_image_h5(file):
    # https://www.getdatajoy.com/learn/Read_and_Write_HDF5_from_Python#Reading_Data_from_HDF5_Files
    import numpy as np
    import h5py
    with h5py.File(file,'r') as hf:
        data = hf.get('Image')
        np_data = np.array(data)
    return np_data

def write_image_h5(file, m1):
    import numpy as np
    import h5py
 
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('Image', data=m1)
    return

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def smooth(y, box_pts):
    import numpy as np
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def OverTh2Array(ArrayIn, Th):
    # return x,y,status array find in ArrayIn when Threshold is passed
    OverTh    = False
    ThArr = []
    
    for i in range (0, len(ArrayIn)):
        if ArrayIn[i]>=Th and not OverTh:
            OverTh     = True
            ThArr.append([i, ArrayIn[i], OverTh])
        if ArrayIn[i]<Th and OverTh:
            OverTh     = False
            ThArr.append([i, ArrayIn[i], OverTh])
    return ThArr

def UnderTh2Array(ArrayIn, Th):
    UnderTh    = False
    ThArr = []

    for i in range (0, len(ArrayIn)):
        if ArrayIn[i]<=Th and not UnderTh:
            UnderTh     = True
            ThArr.append([i, ArrayIn[i], UnderTh])
        if ArrayIn[i]>Th and UnderTh:
            UnderTh     = False
            ThArr.append([i, ArrayIn[i], UnderTh])
    return  ThArr

def swift_root_file(sel, run):
    BASE_URL  = "https://swift.cloud.infn.it:8080/v1/AUTH_1e60fe39fba04701aa5ffc0b97871ed8/Cygnus/"
    file_root = ('Data/'+sel+'/histograms_Run%05d.root' % run)
    return BASE_URL+file_root

def swift_fileByname(sel, filename):
    BASE_URL  = "https://swift.cloud.infn.it:8080/v1/AUTH_1e60fe39fba04701aa5ffc0b97871ed8/Cygnus/"
    file_tif = ('Data/'+sel+'/'+filename)
    return BASE_URL+file_tif

def reporthook(blocknum, blocksize, totalsize):
    import sys
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))
        
def swift_read_root_file(url):
    import ROOT
    import os
    import urllib
    from platform import python_version
    
    tmpname = "./tmp." + str(os.getpid()) + ".root"
    
    if python_version().split('.')[0]=='3':
        urllib.request.urlretrieve(url, tmpname, reporthook)
    else:
        urllib.urlretrieve(url, tmpname, reporthook)

    f  = ROOT.TFile.Open(tmpname);
    os.remove(tmpname)
    return f   

def swift_download_file(url):
    import os
    import urllib
    from platform import python_version
    
    tmpname = "./tmp." + str(os.getpid()) + ".root"
    
    if python_version().split('.')[0]=='3':
        urllib.request.urlretrieve(url, tmpname, reporthook)
    else:
        urllib.urlretrieve(url, tmpname, reporthook)
        
    return tmpname


def rm_file(filein):
    command = '/bin/rm ' + filein
    return os.system(command)

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



def confidence_ellipse_par(x, y):
    import numpy as np
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    if np.sqrt(cov[0, 0] * cov[1, 1]) == 0:
        width = height = -1
        pearson = np.nan
    else:
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        width=ell_radius_x * 2
        height=ell_radius_y * 2
    return width, height, pearson

def cluster_par(xc, yc, image):
    ph = 0.
    dim = xc.shape[0]
    for j in range(0, dim):
        x = int(xc[j])
        y = int(yc[j])
        ph += (image[y,x])
    return ph, dim

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import numpy as np
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    #print (ellipse, transf)
    return ax.add_patch(ellipse), ellipse


def cluster_elips(points):
    import numpy as np
    x0i= np.argmin(points[:,1])
    a0 = points[x0i][1]
    x1i= np.argmax(points[:,1])
    a1 = points[x1i][1]
    y0i= np.argmin(points[:,0])
    b0 = points[y0i][0]
    y1i= np.argmax(points[:,0])
    b1 = points[y1i][0]
    #print (a0, a1, b0, b1, x0i, points[x0i])
    a  = (a1 - a0)/2.
    b  = (b1 - b0)/2.
    x0 = (a1 + a0)/2.
    y0 = (b1 + b0)/2.
    theta = np.arctan((points[x1i][0]-points[x0i][0])/(points[x1i][1]-points[x0i][1]))
    return x0, y0, a , b, theta



# ############################################################################################################################ #
# Clustering
# ############################################################################################################################ #

def NNClustering(points, thC):
# poi: vettore dei punti X,Y nel piano
    
    import numpy as np
    C = np.zeros((len(points), 4))
    for i in range(0,  len(points)):
        C[i]=[i, 0, points[i,1], points[i,0]]
    
    NofC  = 0
    NeC   = 0
    nloop = 0
    while True:
        i = 1
        nordered = 0
        while (i < len(C)):
            j = 0
            while (j < i):
                sBreak = False
                if abs(C[j,2]-C[i,2])<thC and abs(C[j,3]-C[i,3])<thC:   # close point i to j
                    NofCj = C[j,0]
                    NeCj  = (C[C[:,0]==NofCj][:,1]).max()
                    NofCi = C[i,0]
                    NeCi  = (C[C[:,0]==NofCi][:,1]).max()
                    
                    if NofCi != NofCj:
                        if NeCi == 0:
                            C[i,0]= NofCj
                            C[i,1]= NeCj+1
                        else:
                            if NofCi>NofCj:
                                Ci = np.where(C[:,0]==NofCi)
                                for iCi in range(0, len(Ci)):
                                    C[Ci[iCi], 0] = NofCj
                                    C[Ci[iCi], 1] = NeCj+iCi+1
                            else: 
                                Cj = np.where(C[:,0]==NofCj)[0]
                                for iCj in range(0, len(Cj)):
                                    C[Cj[iCj], 0] = NofCi
                                    C[Cj[iCj], 1] = NeCi+iCj+1
                        sBreak = True
                        nordered += 1
                        break
                j += 1
            i +=1
        if nordered == 0:
            break
        nloop += 1

    sorted_C = np.array(sorted(C, key=lambda x:x[0]))
    return sorted_C

# HDBSCAN clustering. Uses HDBSCAN to generate clusters and returns an output matrix formatted as in NNClustering, 
# i.e. each row follows the format [cluster label, intracluster label, yval, xval]. The matrix has dimensions 
# points.shape[0] x 4, i.e. number of points x 4. 
def HDBSClustering(points, min_cluster_size):
    import hdbscan
#     blobs, labels = make_blobs(n_samples=points.shape[0], n_features=2) #centers = 3 is default; number of clusters
#     print 'blobs: ', blobs.size, blobs.shape, labels
#     print points, type(points), points.shape[0], points.shape[1]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(points)
    labels = clusterer.labels_
#     print('labels: ', labels, type(labels), labels.size, labels.shape, labels.shape[0])
#     print('number of labels/clusters: ', clusterer.labels_.max()+1)
    output = np.zeros((len(points), 4))
    intracluster_numbering = {} # dictionary to fill second column of output matrix, i.e. the numbering of a datum within a cluster
    for j in range(-1, clusterer.labels_.max()+1): #include clusterer.labels_.max()
        intracluster_numbering[j] = 0
    for i in range(len(points)):
        output[i]=[labels[i]+1, intracluster_numbering[labels[i]], points[i,1], points[i,0]]
        intracluster_numbering[labels[i]]+=1
    sorted_C = np.array(sorted(output, key=lambda x:x[0]))
    return sorted_C

def clusteringWithMethod(points, min_cluster_size, Cmethod):
    thC = 2         # minimum distanca between points in a cluster (rebinne image)
    if Cmethod == 'hdbs': # nccs
        C = HDBSClustering(points, min_cluster_size)
    elif Cmethod == 'nccs':
        C = NNClustering(points, thC)
    else:
        C = []
    return C

def ClusteringInfo(C):
# ruturn NNClustering clusetr Info array, number of not clusterd, and size of lagre cluster 
    import numpy as np
    NC0  = 0
    NCL  = 0
    maxc = 0
    imax = 0
    info = []
    for i in range(0, len(C)):
        if C[i][1]>0:
            if C[i,1]==1:
                csize = np.where(C[:,0]==C[i,0])[0]
                if len(csize) > maxc:
                    maxc = len(csize)
                    imax = NCL
                info.append(csize)
                NCL +=1
            else:
                NC0 +=1 
    return maxc, imax, info 

# #####################

def PointDistMax(points):
# ruturn the max distance between to point along a line (only a line)
    import numpy as np
    dmax = np.sqrt((points[:,0].max()-points[:,0].min())**2+(points[:,1].max()-points[:,1].min())**2)
    return dmax

def PointDist(p1, p2):
    import numpy as np
    (x1, y1), (x2, y2) = p1, p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ############################################################################################################################ #
# filling 1d and 2d histograms, credits https://github.com/NichtJens/numpy-accumulative-histograms
# ############################################################################################################################ #

class Hist1D(object):

    def __init__(self, nbins, xlow, xhigh):
        self.nbins = nbins
        self.xlow  = xlow
        self.xhigh = xhigh

        self.range = (xlow, xhigh)

        self.hist, edges = np.histogram([], bins=nbins, range=self.range)
        self.bins = (edges[:-1] + edges[1:]) / 2.

    def fill(self, arr):
        hist, _ = np.histogram(arr, bins=self.nbins, range=self.range)
        self.hist += hist

    @property
    def data(self):
        return self.bins, self.hist


class Hist2D(object):

    def __init__(self, nxbins, xlow, xhigh, nybins, ylow, yhigh):
        self.nxbins = nxbins
        self.xhigh  = xhigh
        self.xlow   = xlow

        self.nybins = nybins
        self.yhigh  = yhigh
        self.ylow   = ylow

        self.nbins  = (nxbins, nybins)
        self.ranges = ((xlow, xhigh), (ylow, yhigh))

        self.hist, xedges, yedges = np.histogram2d([], [], bins=self.nbins, range=self.ranges)
        self.xbins = (xedges[:-1] + xedges[1:]) / 2.
        self.ybins = (yedges[:-1] + yedges[1:]) / 2.

    def fill(self, xarr, yarr):
        hist, _, _ = np.histogram2d(xarr, yarr, bins=self.nbins, range=self.ranges)
        self.hist += hist

    @property
    def data(self):
        return self.xbins, self.ybins, self.hist

# #####################

class fillXY:
    def __init__(self):
        self.n = 0
        self.y = np.array([])
        self.x = np.array([])
    def fill(self, x):
        self.n += 1
        self.x = np.append(self.x, self.n)
        self.y = np.append(self.y, x)
    @property
    def data(self):
        return self.x, self.y
    
    
# ############################################################################################################################ #
# Syile (ATLAS-style)
# ############################################################################################################################ #

def set_atlas_style(shape="medium"):
    import matplotlib.pyplot as plt
    """Set the plotting style to ATLAS-style and then point this function to 'None' so that it can only be called once. Called on canvas creation."""

    # Set figure layout
    if shape == "medium":
        plt.rcParams["figure.figsize"] = (10.0, 6.0)
    elif shape == "large":
        plt.rcParams["figure.figsize"] = (20.0, 20.0)
    elif shape == "xlarge":
        plt.rcParams["figure.figsize"] = (30.0, 30.0)
    elif shape == "long":
        plt.rcParams["figure.figsize"] = (20.0, 5.0)
    elif shape == "xlong":
        plt.rcParams["figure.figsize"] = (30.0, 10.0)
    elif shape == "square":
        plt.rcParams["figure.figsize"] = (6, 6)
    elif shape == "two":
        plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["figure.subplot.bottom"] = 0.16
    plt.rcParams["figure.subplot.top"] = 0.95
    plt.rcParams["figure.subplot.left"] = 0.16
    plt.rcParams["figure.subplot.right"] = 0.95

    # Set font options
    #plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = "Helvetica, helvetica, Nimbus Sans L, Mukti Narrow, FreeSans"  # alternatives if helvetica is unavailable
    plt.rcParams["font.cursive"] = "Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, cursive, Helvetica, helvetica"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.default"] = "sf"
    plt.rcParams["mathtext.cal"] = "cursive"
    plt.rcParams["mathtext.bf"] = "sans:bold"
    plt.rcParams["mathtext.it"] = "sans:italic"
    plt.rcParams["mathtext.rm"] = "serif"
    plt.rcParams["mathtext.sf"] = "sans"
    plt.rcParams["mathtext.tt"] = "sans"

    # Set axes options
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["xtick.major.size"] = 12
    plt.rcParams["xtick.minor.size"] = 6
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["ytick.major.size"] = 14
    plt.rcParams["ytick.minor.size"] = 7

    # Set line options
    plt.rcParams["lines.markersize"] = 8
    plt.rcParams["lines.linewidth"] = 1

    # Set legend options
    plt.rcParams["legend.numpoints"] = 1
    plt.rcParams["legend.fontsize"] = 22 #19
    plt.rcParams["legend.labelspacing"] = 0.3
    plt.rcParams["legend.frameon"] = True
    
    # set title dtyle
    plt.rcParams["axes.titlesize"] = 18
    # Disable calling this function again
    #set_atlas_style.func_code = (lambda: None).func_code

####################################
######## HIS to ROOT ###############
####################################
"""
mockNDarray makes a lists of ndarrays (or lists)
            immitate an ndarray (without data copy)
"""
# from __future__ import absolute_import
# __author__  = "Sebastian Haase <seb.haase@gmail.com>"
# __license__ = "BSD license - see LICENSE file"

import numpy as np

mmap_shape = None # default: map entire file; change this to handle BIG files


class mockNDarray(object):
    def __init__(self, *arrs):
        def conv(a):
            if hasattr(a,'view'):
                return a.view()   # we use view(), so that we can fix it up
            if hasattr(a,'__len__'):
                return mockNDarray(*a) # recursively mockify lists
            if a is None:
                return mockNDarray()   # None makes an empty mockNDarray
            
            return np.array(a) # use standard conversion to ndarray (copy!) 
            # raise ValueError, "don't know how to mockify %s" %(a,)
        self._arrs = [conv(a) for a in arrs]
        self._mockAxisSet(0)

    def _mockAxisSet(self, i):
        """
        this is the workhorse function, that makes the internal state consistent
        sets:
           self._mockAxis
           self._ndim
           self._shape
        """
        self._mockAxis = i
        if len(self._arrs)==0:
            self._ndim     = 0
            self._shape    = ()
            return
        self._ndim     = 1+max((a.ndim for a in self._arrs))
        self._shape    = [1]*self._ndim

        #fix up shape of sub array so that they all have ndim=self._ndim-1
        #  fill shape by prepending 1s
        #  unless ndim was 0, then set shape to tuple of 0s
        nd1 = self._ndim-1
        for a in self._arrs:
            if a.ndim == nd1:
                continue
            if isinstance(a, mockNDarray):
                if a._ndim ==0:
                    a._ndim= nd1
                    a._shape = (0,)*nd1
            else:
                if a.ndim == 0:
                    a.shape = (0,)*nd1
                else:
                    a.shape = (1,)*(nd1-a.ndim)+a.shape

        # fixup the shape to reflext the "biggest" shape possible, like a "convex hull"
        iSub=0 # equal to i, except for i>_mockAxis: its one less
        for i in range(self._ndim):
            if i == self._mockAxis:
                self._shape[i] = len(self._arrs)
                continue
            for a in self._arrs:
                # OLD: the a.ndim>iSub check here means, that sub arrays may be "less dimensional" then `self` would imply if it was not "mock"
                # OLD:   if a.ndim <= self._ndim and a.ndim>iSub:
                if self._shape[i] < a.shape[iSub]:
                    self._shape[i] = a.shape[iSub]
            iSub+=1
        self._shape = tuple( self._shape )

    def _getshape(self):
        return self._shape
    def _setshape(self, s):
        # minimal "dummy" implementation
        #  useful for functions like U.mean2d, which want to set shape to -1,s[-2],s[-1]
        __setShapeErrMsg = "mockNDarray supports only trivial set_shape functionality"
        foundMinus=False
        if len(self._shape) != len(s):
            raise ValueError(__setShapeErrMsg)
        for i in range(len(self._shape)):
            if s[i] == -1:
                if foundMinus:
                    raise ValueError( __setShapeErrMsg)
                else:
                    foundMinus = True
            elif s[i] != self._shape[i]:
                    raise ValueError (__setShapeErrMsg)

    shape = property( _getshape,_setshape )


    def _getndim(self):
        return self._ndim
    ndim = property( _getndim )

    def _getdtype(self):
        return min((a.dtype for a in self._arrs)) if self._ndim else None
    dtype = property( _getdtype )

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        import copy
        if isinstance(idx, int):
            if self._mockAxis == 0:
                return self._arrs[idx]
            else:
                s = copy.copy(self)
                s._arrs = [a[idx] for a in self._arrs]
                s._mockAxisSet( self._mockAxis-1 )
                return s
        elif isinstance(idx, tuple):
            if idx == ():
                return self
            if Ellipsis in idx:
                # expand Ellipsis [...] to make slice-handling easier ....
                dimsGiven = len(idx)-1
                for EllipsisIdx in range(len(idx)):
                    if idx[EllipsisIdx] is Ellipsis:
                        break
                idx = idx[:EllipsisIdx ] + (slice(None),)*(self._ndim-dimsGiven) + idx[EllipsisIdx+1:]

            if len(idx) <= self._mockAxis:
                mockIdx = slice(None)
                idxSkipMock = idx
            else:
                mockIdx = idx[self._mockAxis]
                idxSkipMock = idx[:self._mockAxis] + idx[self._mockAxis+1:]
            

            if isinstance(mockIdx, slice):
                s = copy.copy(self)
                
                s._arrs = [a[idxSkipMock] for a in self._arrs[mockIdx]]
                shiftMockAxisBecauseOfInt = sum((1 for i in idx[:self._mockAxis] if not isinstance(i, slice)))
                s._mockAxisSet( self._mockAxis-shiftMockAxisBecauseOfInt )          
                return s
            elif mockIdx is None:
                s = copy.copy(self)
                s._arrs = [a[None][idxSkipMock] for a in self._arrs]
                s._mockAxisSet( self._mockAxis+1 )
                idxSkipMock = (slice(None),)+idxSkipMock  # adjust idxSkipMock to keep new axis 
                return s[idxSkipMock]
                
            else: # mockIdx is "normal" int - CHECK
                # return non-mock ndarray, (or mockNDarray, if there are nested ones)
                return self._arrs[mockIdx][idxSkipMock]

        elif idx is Ellipsis:
            return self
        elif isinstance(idx, slice):
            #raise RuntimeError, "mockarray: slice indices not implemented yet"
            s = copy.copy(self)

            if self._mockAxis ==0:
                s._arrs = self._arrs[idx]
                #s._shape[0] = len(self._arrs)
            else:
                s._arrs = [a[idx] for a in self._arrs[mockIdx]]
                #s._shape[0] = len(self._arrs[0])
            #shiftMockAxisBecauseOfInt = sum((1 for i in idx[:self._mockAxis] if not isinstance(i, slice)))
            s._mockAxisSet( self._mockAxis )
            return s
        elif idx is None: # np.newaxis:
            s = copy.copy(self)
            s._arrs = [a[None] for a in self._arrs]
            s._mockAxisSet( self._mockAxis+1 )
            return s


        raise IndexError ("should not get here .... " )

    def transpose(self, *axes):
        if len(axes) == 1:
            axes = axes[0] # convert  transpose(self, axes) to  transpose(self, *axes)

        if len(axes) != self._ndim:
            raise ValueError ("axes don't match mockarray")

        for newMockAxis in range(self._ndim):
            if axes[newMockAxis] == self._mockAxis:
                break
        else:
            raise ValueError ("axes don't contain mockAxis")

        othersAxes = (ax if ax<newMockAxis else ax-1 for ax in axes[:newMockAxis] + axes[newMockAxis+1:])

        othersAxes = tuple(othersAxes)
        import copy
        s = copy.copy(self)
        s._mockAxisSet(newMockAxis)
        #s._shape = tuple(np.array(s._shape)[list(axes)])

        for i,a in enumerate(s._arrs):
            s._arrs[i] = a.transpose( *othersAxes )

        return s

    def view(self):
        from copy import copy
        return copy(self)
    
    
    
# 64 bytes
dtypeHIS = np.dtype([
    ('magic', 'a2'),
    ('ComLen', np.uint16),
    ('iDX', np.uint16),
    ('iDY', np.uint16),
    ('iX', np.uint16),
    ('iY', np.uint16),
    ('pixType', np.uint16),
    ('numImgs', np.uint32),
    ('numChan', np.uint16),
    ('chan', np.uint16),
    ('timeStamp', np.float64),
    ('marker', np.uint32),
    ('miscinfo', '30i1'),
    ])

hisType2numpyDtype = {
    1: np.uint8,
    2: np.uint16,
    3: np.uint32,
    11: ('RGB', (np.uint8,np.uint8,np.uint8)),
    12: ('RGB', (np.uint16,np.uint16,np.uint16)),
}


class ndarray_inHisFile(np.ndarray):
    def __new__(cls, input_array, hisInfo=None):
        obj = np.asarray(input_array).view(cls)
        obj.HIS = hisInfo
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.HIS = getattr(obj, 'HIS', None)

def _try_openHIS_fastMap(m):
    hisHdr0 = m[:64]
    hisHdr0.dtype = dtypeHIS
    try:
        hisHdr0 = hisHdr0[0]
    except IndexError:
        raise EOFError("zero Bytes HIS file")

    imgPixDType = hisType2numpyDtype[ hisHdr0['pixType'] ]
    pixBytes = imgPixDType().itemsize
    nx,ny,nz = hisHdr0['iDX'],  hisHdr0['iDY'],  hisHdr0['numImgs']
    comLen=hisHdr0['ComLen']

    expectedBytes = (64 + pixBytes*nx*ny) * nz + comLen

#    commentato in python3 deve esseci una diffrenza con python2, va studiato
#    if expectedBytes != len(m):
#        print(expectedBytes, len(m))
#        return None # there are probably comments in other sections, fastMap cannot be used

    mm = m[comLen:] # first hdr will be "corrupt", since comment is just before first imgData
    a = np.recarray(nz, dtype=[( 'hdr',     dtypeHIS ), 
                              ( 'imgData', (imgPixDType, (ny,nx)) ),
                              ], 
                   buf=mm)

    if comLen:
        hisComment = m[64:64+comLen]
        hisComment.dtype = '|S%d'%(comLen,)
    else:
        hisComment = ('',)
    comment = hisComment[0]  # there is "one" comment per sect

    class hisInfo:
        hdr0 = hisHdr0
        comment0 = comment
        hdr = a['hdr']

    fastHisArr = ndarray_inHisFile(a['imgData'], hisInfo=hisInfo)
    return fastHisArr
    
def openHIS(fn, mode='r'):

    """
    open Hamamatsu Image Sequence
    return a mockNDarray
    each section contains a HIS attribute,
        which contains hdr, offsetNext and comment
    """
    m = np.memmap(fn, shape=mmap_shape, mode=mode)

    if mmap_shape is None:
        a = _try_openHIS_fastMap(m)
        if a is not None:
            return a

    offset=0
    imgs = []
    while 1: # for i in range(10):
        try:
            img = readSection(m, offset)
        except:
            break
        imgs.append(img)

        offset = img.HIS.offsetNext

    return mockNDarray(*imgs)

def write2root(fname, img, id=0, option='update'):
    import ROOT
    tf = ROOT.TFile.Open(fname+'.root', option)
    (nx,ny) = img.shape
    h2 = ROOT.TH2D('pic_run',fname+'_'+str(id),nx,0,nx,ny,0,ny)
    h2.GetXaxis().SetTitle('x')
    h2.GetYaxis().SetTitle('y')
    [h2.SetBinContent(bx,by,img[bx,by]) for bx in range(nx) for by in range(ny)]
    h2.Write()
    tf.Close()
