""" come fare i plot con jupyter QtConsole su run `jupyter qtconsole`
import numpy as np
import matplotlib.pyplot as plt
%pylab inline o %pylab se uno lo vuole nel tool QT.
x=np.linspace(0,2*np.pi,100)
y=np.sin(x)*np.exp(-x)
plt.plot(x,y)
%history
import mylib
reload(mylib)
-----------------------------------
comadi bash pre analisi mu rate 
grep Start stat1* | awk '{print $4" "$5 $6 $7 $8}' > mu1.dat
grep muons stat1* | awk '{print $3" "$4" "$5}' > mu2.dat
paste -d " " mu1.dat mu2.dat > muons.dat
days, times, mu1, mu2, mu3 = loadtxt("muons.dat", unpack=True, converters={ 0: matplotlib.dates.strpdate2num('%d-%b-%Y'), 1: matplotlib.dates.strpdate2num('%H:%M:%S.%f')})
plot_date(days, mu3)
filter = (mu3>3500)
plot_date(times[filter], mu3[filter], linestyle='none')
-------------------------------
LOCALITA, DATA, TMEDIA, TMIN, TMAX, PUNTORUGIADA, UMIDITA, VISIBILITA, VENTOMEDIA, VENTOMAX, RAFFICA, PRESSIONESLM, PRESSIONEMEDIA, PIOGGIA, FENOMENI = loadtxt("meteo/2014/Frascati-2014-Settembre.csv", unpack=True, dtype=str, delimiter=';', skiprows=1, converters={ 1: matplotlib.dates.strpdate2num('%d/%m/%Y'), 2: lambda x : x.strip('"'), 11: lambda x : x.strip('"')})
plot_date(numpy.array(DATA, dtype=numpy.float64), PRESSIONESLM)
hist(numpy.array(PRESSIONESLM, dtype=numpy.float64))
--------------------------------
per pulire  plot
cla()
clf()

mylib.write_image_h5("prova.h5", diff)
np.savetxt('test.out', data, delimiter=' ')
data = np.loadtxt('test.out', delimiter=' ', skiprows=1)
from glue import qglue
qglue(xy=vect)

"""
import numpy as np

def purge_apice(x):
    for i in range(0,len(x)):
        x[i]=float(x[i][1:len(x[i])-1])
    return x

def read_datafile(file_name):
    import numpy as np
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=' ', skiprows=10)
    return data

def read_cvs_file(filename):
    import csv
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data_list = map(tuple, reader)
    return data_list

def file_header(data):
    # return line string value
    sval=data[0]
    return sval

def colum_index(sval,data):
    # return colum index of string header value
    ind=data[0].index(sval)
    return ind

def data_col(ind,data):
    # return float array of a given colum by a given index
    val = []
    for i in range(1,len(data)):
        val.append(float(data[i][ind]))
    return val

def data_xy(x,y,data):
    # return float matrix of a given xy
    valx = []
    valy = []
    for i in range(1,len(data)):
        valx.append(float(data[i][x]))
        valy.append(float(data[i][y]))
    val = [valx,valy]
    return val

def k2eV(x):
# 1 degree kelvin= 8.621738 X10-5 eV
    eV = x*8.6217e-5
    return eV
    
def eV2k(x):
# 1 degree kelvin= 8.621738 X10-5 eV
    k = x/8.6217e-5
    return k
# 
def DEbarra(dWdx, z, l, theta):
    import numpy as np
    c0 = 4/(9*np.pi)  
    gamma = 1000.0  # grnusien 
    rho = 2700      # densita' alluminio kg/m^3
    L = 3.0         # lunghezza barra in m
    v = 5400.0      # velocita del suono m/s
    R = 0.3         # raggio barra m
    c1 = (gamma**2)/(rho*L*v**2)
    geofac = np.sin(np.pi*z/L)*((np.sin(np.pi*l*np.cos(theta)))/((2*L)/(np.pi*R*np.cos(theta)/L)))
    DE = c0*c1*(dWdx**2)*geofac
    return DE

# funzione approssimata per particelle verticali
def DEb0(W): # W GeV
    """For a particle that loses a total amount of energy W while intersecting 
    the bar, orthogonally to its axis and through its middle section, 
    the energy of the longitudinal fundamental mode of vibra- tion will 
    change by
    """
    dG = 1.3
    """Al5056 alloy, give: dG,n =1.16 for T=4K and dG,n=1.3 for T~1.5K
    abovethe transition temperature ( Tc = 0.845 K), with a relative 
    uncertainty of 6%. For superconductive Al5056 we found dG,s = 5.7 +/- 0.9.
    """
    DE = 7.64e-9 * W**2 * dG**2
    return DE # in Kelvin

def EG(omega): #
    import numpy
    G = 6.67259e-11
    c = 299792458
    f = 1e-21
    E = (c**2/(32*numpy.pi*G)) * omega**2 * f**2
    return E

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

def hist(list, title):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab

    plt.figure(1)
    plt.hist(list, normed=True)
    plt.xlim((min(list)-1, max(list)+1))

    mean = np.mean(list)
    variance = np.var(list)
    sigma = np.sqrt(variance)
    x = np.linspace(min(list), max(list), 100)
    #plot
    plt.xlabel('Bins')
    plt.ylabel('% / Counts')
    plt.title(r'$\mathrm{%s -}\ \mu=%.3f,\ \sigma=%.3f$' %(title, mean, sigma))
    plt.grid(True)
    
    plt.plot(x,mlab.normpdf(x,mean,sigma))
    
    plt.show()
    return

def gfit(list, bins):

    from scipy.optimize import leastsq
    from scipy.interpolate import spline
    import numpy as np
    from scipy.stats import chisquare

    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))

    hist, _  = np.histogram(list, bins=bins)
    xdata    = np.linspace(np.min(list), np.max(list), bins)
    ydata    = hist
    mean     = np.mean(list)
    variance = np.var(list)
    sigma    = np.sqrt(variance)
    init     = [1.0, mean, sigma, 0.0]
    
    out      = leastsq( errfunc, init, args=(xdata, ydata))
    c        = out[0]

    x_smooth = np.linspace(np.min(list), np.max(list), bins)
    y_smooth = spline(xdata, fitfunc(c, xdata), x_smooth)
    chi = chisquare(ydata, f_exp=y_smooth)
    
#    print "A exp[-0.5((x-mu)/sigma)^2] + k "
#    print "Fitings Coefficients:"
#    print ("lastsq fit A = %.3f mu = %.3f sigma = %.3f k = %.3f" % (c[0],c[1],abs(c[2]),c[3]))
#    print ("Chi2 = %.2f p = %.2f" % (chi[0],chi[1]))
    return x_smooth, y_smooth, c, chi

def gfit3(list, bins):

    from scipy.optimize import leastsq
    from scipy.interpolate import spline
    import numpy as np
    from scipy.stats import chisquare

    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))

    hist, _  = np.histogram(list, bins=bins)
    xdata    = np.linspace(np.min(list), np.max(list), bins)
    ydata    = hist
    mean     = np.mean(list)
    variance = np.var(list)
    sigma    = np.sqrt(variance)
    init     = [1.0, mean, sigma]
    
    out      = leastsq( errfunc, init, args=(xdata, ydata))
    c        = out[0]

    x_smooth = np.linspace(np.min(list), np.max(list), bins)
    y_smooth = spline(xdata, fitfunc(c, xdata), x_smooth)
    chi = chisquare(ydata, f_exp=y_smooth)
    
#    print "A exp[-0.5((x-mu)/sigma)^2] + k "
#    print "Fitings Coefficients:"
#    print ("lastsq fit A = %.3f mu = %.3f sigma = %.3f k = %.3f" % (c[0],c[1],abs(c[2]),c[3]))
#    print ("Chi2 = %.2f p = %.2f" % (chi[0],chi[1]))
    return x_smooth, y_smooth, c, chi

def read_h5_write_txt(fileH5, fileDAT):
    import sys
    import numpy as np
    image = read_image_h5(fileH5)
    np.savetxt(fileDAT, image, fmt='%d', delimiter='\t')
    return

def filegerp(file, str):
    import re
    for line in open(file, 'r'):
        if re.search(str, line):
            return line
        if line == None:
            return ''
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

def Gauss(x, a0, x0, s0, y0):
    import numpy as np
    return a0 * np.exp(-(x - x0)**2 / (2 * s0**2)) + y0

def Gauss3(x, a0, x0, s0):
    import numpy as np
    return a0 * np.exp(-(x - x0)**2 / (2 * s0**2))


def Line(x, m, q): # this is your 'straight line' y=f(x)
    import numpy as np
    return m*x + q

def InvX(x, p0, p1): # uno su X

    import numpy as np
    return p0/x +p1

def Exp(x, p0, p1, p2): # negative exponential

    import numpy as np
    return p0*np.exp(-x/p1)+p2

def ExpGain(x, p0, p1): # power of ten

    import numpy as np
    return p0*np.exp(p1*x)

def ExpFD(x, p0, p1, p2, p3): # negative exponential + fermi dirac rise
    # (1/(1+exp(([0]-x)/[1])))*(-[2]*exp(-x/[3])) # Natalia/Pinci 
    import numpy as np
    return (1/(1+np.exp((p0-x)/p1)))*(-p2*np.exp(-x/p3))

def GExp(x, p0, p1, p2, p3, p4): # Gauss + expnential tail
    import numpy as np
    return p0*(np.exp(-(x - p1)**2 / (2 * p2**2)) + p3*(-np.exp(-x/p4)))

def TwoExp(x, p0, p1, p2, p3): # Gauss + expnential tail
    import numpy as np
    return p0*(-np.exp(-x/p1)) + p2*(-np.exp(-x/p3))

def smooth(y, box_pts):
    import numpy as np
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def IntegraArr(arr, bins):
    import numpy as np
    y = []
    x = []
    slide = np.int(np.size(arr)/bins)
    e_slide = 0
    for i in range(0, bins):
        s_slide = e_slide 
        e_slide = s_slide + slide
        y.append(np.sum(arr[s_slide:e_slide]))
        x.append(i)
    return x, y
 
def online_maen_var(data):
    n = 0
    mean = 0.0
    M2 = 0.0
     
    for x in data:
        n += 1
        delta = x - mean
        mean += delta/n
        M2 += delta*(x - mean)

    if n < 2:
        return float('nan')
    else:
        return mean, M2 / (n - 1)

# #####################
# Syile (ATLAS-style)
# #####################

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
    plt.rcParams["font.family"] = "sans-serif"
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
    plt.rcParams["legend.fontsize"] = 19
    plt.rcParams["legend.labelspacing"] = 0.3
    plt.rcParams["legend.frameon"] = True
    
    # set title dtyle
    plt.rcParams["axes.titlesize"] = 18
    # Disable calling this function again
    #set_atlas_style.func_code = (lambda: None).func_code

# #####################


# #####################
# Clustering
# #####################

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

def NNClusteringInfo(C):
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

# #####################
# filling 1d and 2d histograms, credits https://github.com/NichtJens/numpy-accumulative-histograms
# #####################
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
