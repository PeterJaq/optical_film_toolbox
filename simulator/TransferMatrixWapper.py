# -*- coding: utf-8 -*-
"""
@author: C. Marcus Chuang, 2015
"""
import os 
import sys
os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append("/home/peterjaq/project/optical-film-maker/")
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import bisect
plt.style.use('ggplot')

from common.DataLoader import MaterialLoader

# Constants
h = 6.626e-34  # Js Planck's constant
c = 2.998e8  # m/s speed of light
q = 1.602e-19  # C electric charge


class OpticalModeling(object):
    def __init__(self, Materials,
                       WLrange = (350, 1200),
                       WLstep  = 1.0,
                       posstep = 1.0,
                 ):
        """
        Initialize an OpticalMpdeling instance, load required data,
        and initialize required attributes
        """
        # load material list
        self.material_li = [x for x in Materials if x != 'Air']

        # material Loader 
        self.material_loader = MaterialLoader()
        self.material        = self.material_loader.load_select_material(self.material_li)

        # layers (materials)
        self.layers = Materials
        # thickness; set the thickness of the substrate (first layer) to 0
        # self.t = [0] + [mat[1] for mat in Device[1:] if float(mat[1]) > 0]
        # self.t_cumsum = np.cumsum(self.t)
        # wavelength
        self.WL      = np.arange(WLrange[0], WLrange[1], WLstep)
        self.WLstep  = WLstep
        self.posstep = posstep
        # positions to evaluate field

        self.AM15 = self.LoadSolar("./db/SolarAM15.csv")  # load/reshape into desired WLs
        self.nk   = self.load_nk(materials=self.material)

        # dp for calculating transfer matrices (used in CalE() )
        self.Imats = {}
        self.Lmats = {}

        """
        Below are some attributes in the instance. They don't have to be
        initialzed here, but doing so (setting them to None) makes it somewhat
        easier to keep track of the attributes
        """

        # ## These will be 2D numpy arrays, row: position; column: wavelength
        self.E       = None  # E field
        self.AbsRate = None  # Absorption rate
        self.Gx      = None  # carrier generation rate

        # ### These will be 1D numpy arrays, x: position
        self.Reflection       = None  # Reflection
        self.Transmission     = None  # Transmission
        self.Absorption       = None # Absorption
        self.total_Absorption = None
        self.mean_abs         = None

        # ### This will be a 2D pandas dataframe,
        # wavelegths vs absorption for each material
          # Absorption
        self.Absorption_layer = None

        self.Jsc = None  # 1D array, Max Jsc in each layer

    def RunSim(self, thickness):
        """
        Run the simulation. This method would complete all the caculation.
        """

        self.t        = thickness
        self.t_cumsum = np.cumsum(self.t)

        self.x_pos    = np.arange(self.WLstep / 2.0, sum(self.t), self.posstep)
        self.x_ind    = self.x_indice()

        self.Cal_Imat_Lmats()
        S, Sp, Sdp = self.CalS()
        self.CalE(S, Sp, Sdp)
        self.CalAbs()
        self.CalGen()

        return None

    def x_indice(self):
        """
        return a list of indice "x_ind" for use in x_pos
        material i corresponds to the indices range
        [x_ind[i-1], xind[i]) in x_pos
        Note: the first layer is glass and is excluded in x_pos
        """
        return [bisect.bisect_right(self.x_pos, self.t_cumsum[i])
                for i in range(len(self.t))]

    def CalE(self, S, S_prime, S_dprime):
        """
        Calculate incoherent power transmission through substrate
        T = |4*n1*n2 / (n1+n2)^2| , R = |((n1-n2)/(n1+n2))^2|
        It would calculate and update
        1. The electric field in the device stack (self.E)
        2. The reflection (self.Reflection)
        """

        subnk = self.nk[self.layers[0]]

        T_glass = abs(4 * 1 * subnk / (1 + subnk)**2)
        R_glass = abs(((1 - subnk) / (1 + subnk))**2)

        # Calculate transfer matrices and field at each wavelength and position
        self.E = np.zeros([len(self.x_pos), len(self.WL)], dtype=complex)

        R = (abs(S[:, 1, 0] / S[:, 0, 0]))**2
        T = abs(2.0 / (1 + self.nk[self.layers[0]])) / (1 - R_glass * R)**0.5

        layers = self.layers + ['Air']
        for matind in range(1, len(layers) - 1):  # last one is 'air', ignore
            mater = layers[matind]
            for i, w in enumerate(self.WL):
                xi = 2.0 * np.pi * self.nk[mater][i] / w
                dj = self.t[matind]

                # x: distance from previous layer
                x = (self.x_pos[self.x_ind[matind - 1]:self.x_ind[matind]] -
                     self.t_cumsum[matind - 1])

                Sp, Sdp = S_prime[matind][i], S_dprime[matind][i]

                numerator = (
                    Sdp[0, 0] * np.exp(complex(0, -1.0) * xi * (dj - x)) +
                    Sdp[1, 0] * np.exp(complex(0, 1.0) * xi * (dj - x)))

                denom = (
                    Sp[0, 0] * Sdp[0, 0] * np.exp(complex(0, -1.0) * xi * dj) +
                    Sp[0, 1] * Sdp[1, 0] * np.exp(complex(0, 1.0) * xi * dj))

                l, r = self.x_ind[matind - 1], self.x_ind[matind]
                self.E[l:r, i] = T[i] * numerator / denom

        self.Reflection = R_glass + T_glass**2 * R / (1 - R_glass * R)


        return None

    def Cal_Imat_Lmats(self):
        Imats, Lmats = self.Imats, self.Lmats
        layers = self.layers + ["Air"]
        # precalculate all the required Imat and Lmat
        for matind in range(len(layers)-1):
            mater, nex = layers[matind], layers[matind+1]
            if matind not in Lmats:
                Lmats[matind] = self.L_mat(matind)
            if (mater, nex) not in Imats:
                Imats[(mater, nex)] = self.I_mat(mater, nex)
        return

    def CalS(self):
        '''
        calculate S, S_prime, and S_dprime
        S = S' * L  * S"   for any j
             j    j    j

                        i = j-1
           S'[j]  = [  product  ( I      * L    )  ]   * I
          (j>0)         i = 0      i,i+1    i+1           j, j+1

        '''

        Imats, Lmats = self.Imats, self.Lmats
        nWL = len(self.WL)
        S_prime, S_dprime = {}, {}

        layers = self.layers + ["Air"]

        # calculate S_prime and S
        S = np.array([np.eye(2, dtype=complex) for _ in range(nWL)])
        for matind in range(1, len(layers)):
            pre, mater = layers[matind - 1], layers[matind]
            for i in range(nWL):
                S[i] = S[i].dot(Lmats[matind - 1][i])
                S[i] = S[i].dot(Imats[(pre, mater)][i])
            S_prime[matind] = np.copy(S)

        S_dprime[len(layers)-2] = Imats[(layers[-2], layers[-1])]

        for matind in range(len(layers)-3, 0, -1):
            mater, nex = layers[matind], layers[matind + 1]
            tmp = np.copy(S_dprime[matind + 1])
            for i in range(nWL):
                tmp[i] = np.dot(Lmats[matind+1][i], tmp[i])
                tmp[i] = np.dot(Imats[(mater, nex)][i], tmp[i])
            S_dprime[matind] = tmp

        return S, S_prime, S_dprime

    def CalAbs(self):
        """
        Calculate normalized intensity absorbed /cm3-nm at each position and
        wavelength as well as the total reflection expected from the device
        """
        # Absorption coefficient in cm^-1 (JAP Vol 86 p.487 Eq 23)
        a = pd.DataFrame()
        for mater in self.layers[1:]:
            if mater not in a:
                a[mater] = 4 * np.pi * self.nk[mater].imag / (self.WL * 1e-7)

        # initialize Absrate with E^2, multiply nk later
        self.AbsRate = abs(self.E)**2
        self.Absorption = pd.DataFrame()  # initialize Absorption
        for matind in range(1, len(self.t)):
            mater = self.layers[matind]
            posind = self.x_ind[matind-1], self.x_ind[matind]
            mlabel = "L" + str(matind) + "_" + mater
            self.AbsRate[posind[0]:posind[1]] *= (
                a[mater] * np.real(self.nk[mater])).values
            self.Absorption[mlabel] = (
                np.sum(self.AbsRate[posind[0]:posind[1]], 0) *
                self.posstep * 1e-7)
        self.Transmission = 1.0 - np.sum(self.Absorption, 1) - self.Reflection
        self.total_Absorption = np.sum(self.Absorption, 1)

        return None

    def CalGen(self):
        """
        Calculate generation rates as a function of position in the device
        and calculates Jsc (in mA/cm^2)
        """
        # Energy dissipation mW/cm3-nm at each position and wavelength
        # (JAP Vol 86 p.487 Eq 22)
        if self.AbsRate is None:
            self.CalAbs()
        Q = self.AbsRate * self.AM15
        self.Gx = Q * 1e-12 / (h * c) * self.WL

        Gx_x = [np.sum(self.Gx[self.x_ind[i-1]:self.x_ind[i]])
                for i in range(1, len(self.layers))]
        self.Jsc = np.array(Gx_x) * self.WLstep * self.posstep * q * 1e-4

        return None

    def I_mat(self, mat1, mat2):
        """
        Calculate the transfer matrix I for Reflection and Transmission
        at an interface between two materials.
        mat1, mat2: name of the materials
        return I, a  numpy array with shape len(self.WL)x2x2
        I[i] is the transfer matrix at wavelength self.WL[i]

        I[i] = 1/T[i] *  [ [    1,  R[i] ]
                           [ R[i],     1 ] ]
        """
        n1s, n2s = self.nk[mat1], self.nk[mat2]  # complex dielectric constants
        R = (n1s-n2s) / (n1s+n2s)
        T = 2.0 * n1s / (n1s+n2s)
        I = np.array([[[1.0, R[i]], [R[i], 1.0]] / T[i]
                      for i in range(R.shape[0])])
        return I

    def L_mat(self, matind):
        """
        Calculate the propagation matrix L, through a material
        matind: index of the material
        material name : mat = self.layers[matind]
        thickness     : d   = self.t[matind]
        complex dielectric constants:  self.nk[mat]

        return L, a numpy array with shape len(self.WL)x2x2 array
        L[i] is the propogation matrix at wavelength self.WL[i]

        L[i] = [ [ exp(-x*d),        0 ]
                 [         0, exp(-x*d)] ]
        where x = n*cos(phi)* 2*(pi)/(lambda),
        (n:complex, phi:incident angle, here phi= 0
        """

        mat, d = self.layers[matind], self.t[matind]  # d: thickness
        x = 2.0 * (np.pi) * self.nk[mat] / self.WL
        L = np.array([[[np.exp((-1j) * x[i] * d), 0],
                       [0, np.exp(1j * x[i] * d)]]
                      for i in range(x.shape[0])])
        return L

    def LoadSolar(self, Solarfile):
        Solar = pd.read_csv(Solarfile, header=0)
        AM15 = np.interp(self.WL, Solar.iloc[:, 0], Solar.iloc[:, 1])
        return AM15  # mW/cm2 nm
        
    def load_nk(self, materials):
        """[summary]
        load n and k parameter from the materials

        Arguments:
            materials {[type]} -- [description]
        """
        rtn_nk = {}

        max_wl = 9999
        min_wl = 0
        for mat, mat_data in materials.items():
            max_wl = min(max(mat_data['wl']), max_wl)
            min_wl = max(min(mat_data['wl']), min_wl)

        rtn_nk["Air"] = np.array([1] * len(self.WL))

        for mat in self.layers:
            if mat not in rtn_nk:
                n = np.interp(self.WL, materials[mat]['wl'], materials[mat]['n'])
                k = np.interp(self.WL, materials[mat]['wl'], materials[mat]['k'])
                rtn_nk[mat] = n + 1j * k

        return rtn_nk

    @property
    def simulation_result(self):
        #print(self.total_Absorption, self.Transmission, self.Reflection)
        return self.total_Absorption, self.Transmission, self.Reflection



if __name__ == "__main__":

    Demo = True  # set Demo to True to run an example simulation

    if Demo: 
        Device = [
            "SiO2_Malitson",
            "Zn_Werner",
            "SiO2_Malitson",
            # "Zn_Werner",
            "Cu_Johnson"
            ]
        #libname = "./old_work/data/Index_Refraction_Zn0.16+SiO2.csv"
        OM = OpticalModeling(Device, 
                             WLrange=(350, 1200))
        OM.RunSim(thickness=[79, 32, 81, 100])
        x = np.arange(350, 1200, 1)

        print(f'优化过程中的状态: [吸收]{np.mean(OM.simulation_result[0])}, \
                                [投射]{np.mean(OM.simulation_result[1])}, \
                                [反射]{np.mean(OM.simulation_result)}')

        plt.plot(x, OM.simulation_result[0], 'r--',
                x, OM.simulation_result[1], 'bs',
                x, OM.simulation_result[2], 'g--')
                
        plt.show()
