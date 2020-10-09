# -*- coding: utf-8 -*-
"""
@author: C. Marcus Chuang, 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from TransferMatrix import OpticalModeling as TM
#from optical_film.TransferMatrix import OpticalModeling as TM
plt.style.use('ggplot')


class OMVaryThickness(TM):

    def __init__(self, *args, **kwargs):
        TM.__init__(self, *args, **kwargs)
        self.S_prime = {}
        self.S_dprime = {}

    def set_t_update(self, layerind, newthickness):
        """
        set the thickness of the layer "layerind" and then update
        relevant parameters
        """
        self.t[layerind] = newthickness
        self.t_cumsum = np.cumsum(self.t)
        self.x_pos = np.arange(self.WLstep / 2.0, sum(self.t), self.posstep)
        self.x_ind = self.x_indice()
        self.Lmats[layerind] = self.L_mat(layerind)

        return None

    # def RunSim(self, plotE=True, plotAbs=True, plotGen=True,
    #                 saveFigE=False, saveFigAbs=False, saveFigGen=False):
    def RunSim(self):
        self.Cal_Imat_Lmats()
        S = self.CalS(L_vary=self.L_vary)
        self.CalE(S, self.S_prime, self.S_dprime)
        self.CalAbs()
        self.CalGen()

        return None

    def VaryOne(self, L_vary, t_range, target, toPrint=False,
                PlotJsc=True, PlotAbs=True, cbarlegend=False):
        """
        vary the thickness of the layer with index L_vary and then run optical
        modeling.
        L_vary: index of the layer to vary
        t_range: thickness to vary, an iterable
        self.varyJsc: a list of "Jsc in each layer" w.r.t the varying thickness
        self.varyAbsorption: a list of "absorption in each layer"
        return None
        """
        self.varyJsc = []
        self.varyAbs = []
        self.varyT = []
        self.varyR = []
        self.L_vary = L_vary
        self.t_range = sorted(t_range)
        if not isinstance(target, str):
            target = int(target)  # in case user input is a float
        else:
            target = target.upper()  # in case user input is a lowercase letter

        for ti in self.t_range:
            self.set_t_update(L_vary, ti)  # update t & related variables
            self.RunSim()
            self.varyJsc.append(self.Jsc)
            self.varyAbs.append(self.Absorption)
            self.varyT.append(self.Transmission)
            self.varyR.append(self.Reflection)
            if toPrint:
                print("calculating: ", self.layers[L_vary], "=", ti, "nm,")
                # print self.layers[L_vary], "=", ti, "nm,", "Max Jsc in",
                # print self.layers[target], "=",
                # print np.round(self.Jsc[target - 1], 2)
        if PlotJsc:
            self.PlotVaryJsc(target)
        if PlotAbs:
            self.PlotVaryAbs(target, cbarlegend)

        return None

    def PlotVaryJsc(self, target):
        tmax = len(self.layers) - 1
        errortxt = ("Invalid target index for PlotVaryJsc:\n" +
                    "Target index should be between 1 and {}".format(tmax))
        try:  # catch wrong input
            x = float(target)
            if not int(x) == x or x <= 0 or x > tmax:
                raise ValueError
            target = int(x)
        except:
            if target.lower() == 'all':
                for tar in range(1, tmax + 1):
                    self.PlotVaryJsc(tar)
            else:
                print(errortxt)
            return

        ftitle = 'Max Jsc in L' + str(target) + ' ' + self.layers[target]
        figJsc = plt.figure(ftitle)
        figJsc.clf()
        axJsc = figJsc.add_subplot(111)
        xlabel = 'Thickness of ' + self.layers[self.L_vary] + ' (nm)'
        ylabel = "Jsc from {} (mA/cm$^2$)".format(self.layers[target], size=22)
        axJsc.set_xlabel(xlabel, size=16)
        axJsc.set_ylabel(ylabel, size=16)
        axJsc.plot(self.t_range, [Jsc[target - 1] for Jsc in self.varyJsc],
                   '-o', linewidth=2, color='r', markersize=8)
        tmin, tmax = self.t_range[0], self.t_range[-1]
        mid, width = (tmin + tmax) / 2.0, (tmax - tmin) / 2
        axJsc.set_xlim(xmin=max(0, mid - width * 1.05),
                       xmax=mid + width * 1.05)
        axJsc.tick_params(labelsize=14)
        figJsc.tight_layout()

        return None

    def PlotVaryAbs(self, target, cbarlegend=False):
        """
        plot the absorption in the target layer or the total absorption,
        reflection, or the transmision in the whole device stack
        target can be one of the followings:
        (1) index of the layer of interest
        (2) 'R' or 'reflection' for reflection
        (3) 'T' or 'transmission' for transmission
        (4) 'A' or 'Abs' or 'Absorption' for the total absorption
        (5) 'all' to plot all figures listed above
        """

        tmax = len(self.layers) - 1
        errortxt = (
        "Invalid target for PlotVaryJsc(target): \n" +
        "'target' should be one of the followings: \n" +
        "(1) An index between 1 and " +
             "{0} for the absorption in that layer \n".format(tmax) +
        "(2) 'R' or 'Reflection' (not case sensitive) for reflection \n" +
        "(3) 'T' or 'Transmission' (not case sensitive) for transmission \n"+
        "(4) 'A' or 'Abs' or 'Absorpton' (not case sensitive) for " +
             "the total absorption (1-R-T) \n" +
        "(5) 'all' to plot all figures listed above")

        valid = {'r', 'reflection', 't', 'transmission', 'a', 'abs',
                 'absorption', 'all'}

        try:  # catch wrong input
            x = float(target)
            if not int(x) == x or x <= 0 or x > tmax:
                raise ValueError
            target = int(x)

        except:
            if not isinstance(target, str) or target.lower() not in valid:
                print(errortxt)
                return
            target = target.lower()

        if target == 'all':
            alltar = range(1, tmax + 1) + ['r', 't', 'a']
            for tar in alltar:
                self.PlotVaryAbs(tar, cbarlegend=cbarlegend)
            return

        # figAbs = plt.figure('absorption', figsize=(16*0.8, 9*0.8))
        figAbs = plt.figure(figsize=(16 * 0.8, 9 * 0.8))
        figAbs.clf()
        axAbs = figAbs.add_subplot(111)
        axAbs.set_xlabel('Wavelength (nm)', size=22)

        cmap = plt.get_cmap('rainbow')
        num_color = len(self.t_range)
        vmin, vmax = self.t_range[0], self.t_range[-1]
        normalize = mcolors.Normalize(vmin, vmax)

        if target in ['r', 'reflection']:
            axAbs.set_ylabel('Reflection (%)', size=22)
            for ind, t in enumerate(self.t_range):
                axAbs.plot(self.WL, 100.0 * self.varyR[ind],
                           linewidth=2, label=str(t) + " nm",
                           color=cmap(normalize(t)))

        elif target in ['a', 'abs', 'absorption']:
            axAbs.set_ylabel('Absorption (1-R-T) (%)', size=22)
            for ind, t in enumerate(self.t_range):
                axAbs.plot(self.WL,
                           100.0 * (1 - self.varyR[ind] - self.varyT[ind]),
                           linewidth=2, label=str(t) + " nm",
                           color=cmap(normalize(t)))

        elif target in ['t', 'transmission']:
            axAbs.set_ylabel('Transmission (%)', size=22)
            for ind, t in enumerate(self.t_range):
                axAbs.plot(self.WL, 100.0 * self.varyT[ind],
                           linewidth=2, label=str(t) + " nm",
                           color=cmap(normalize(t)))
        else:
            targethead = self.Absorption.columns[target - 1]
            axAbs.set_ylabel(
                'Absorption in {0} (%)'.format(self.layers[target]), size=22)

            for ind, t in enumerate(self.t_range):
                axAbs.plot(self.WL, 100.0 * self.varyAbs[ind][targethead],
                           linewidth=2, label=str(t) + " nm",
                           color=cmap(normalize(t)))

        # axAbs.set_xlim(xmin = self.WL[0], xmax = self.WL[-1])
        axAbs.tick_params(labelsize=18)
        # use normal legend
        if num_color <= 20 and not cbarlegend:
            axAbs.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         numpoints=1, fontsize=14,
                         title='Thickness of\n ' + self.layers[self.L_vary],
                         borderaxespad=0).draggable()
            axAbs.get_legend().get_title().set_fontsize(16)
            plt.tight_layout()

            figAbs.subplots_adjust(right=0.82)

        # use colorbar legend when specified or >20 lines
        else:
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(self.t_range)
            cblegend = plt.colorbar(scalarmappaple)
            cblabel = ('Thickness of ' +
                       self.layers[self.L_vary] + " (nm)")

            cblegend.set_label(cblabel, fontsize=18, labelpad=16)
            cblegend.ax.tick_params(labelsize=14)
            plt.tight_layout()

        return None

    def VaryTwo(self, L1, t1_range, L2, t2_range, target1, target2=None,
                toPlot=True, print1=False, print2=False,
                interp_countour=False):
        """
        vary the thickness of the two layers L1 and L2.
        L1, L2: indice of the layers to vary
        t1_range, t2_range: thickness to vary, an iterable
        target1: the layer used to calculate Jsc (@100%IQE)
        target2: Only used for tandem cells, where target1 and target2
                 are the two absorber. Calculate the min of (Jsc1, Jsc2),
                 i.e. the current limiting Jsc.
                 Default is None (single junction cells)
        The result will be stored in self.v2Jsc, a 3-D np array

        return None
        """
        v2Jsc = []
        self.L1, self.t1range = L1, sorted(t1_range)
        self.L2, self.t2range = L2, sorted(t2_range)
        for t1 in self.t1range:
            self.set_t_update(self.L1, t1)
            self.S_prime, self.S_dprime = {}, {}
            if print1:
                print("Calculating: ", self.layers[L1], "=", t1, "nm,",)
                print("Varying ", self.layers[L2])
            self.VaryOne(self.L2, self.t2range, target1, toPrint=print2,
                         PlotJsc=False, PlotAbs=False)
            v2Jsc.append(self.varyJsc)
        self.v2Jsc = np.array(v2Jsc)
        if toPlot:
            self.PlotTwo(target1, target2, interp_countour)

        return None

    def PlotTwo(self, target1, target2=None, interp_contour=False):

        if target2 is not None:  # for tandem
            p, q = len(self.t1range), len(self.t2range)
            Jsc = self.v2Jsc
            J = np.array([[min(Jsc[i][j][target1 - 1], Jsc[i][j][target2 - 1])
                           for i in range(p)] for j in range(q)])

            V2title = ("Max Jsc in the device\n (min of L{0} {1} and L{2} {3})"
                       ).format(target1, self.layers[target1],
                                target2, self.layers[target2])
        else:
            J = self.v2Jsc.take(target1 - 1, axis=2).T
            V2title = 'Max Jsc in L{0} {1}'.format(
                target1, self.layers[target1])

        figV2 = plt.figure(V2title)
        figV2.clf()
        axV2 = figV2.add_subplot(111)
        xlabel = 'Thickness of L' + str(self.L1) + ' ' + self.layers[self.L1]
        ylabel = 'Thickness of L' + str(self.L2) + ' ' + self.layers[self.L2]
        axV2.set_xlabel(xlabel + ' (nm)', size=14)
        axV2.set_ylabel(ylabel + ' (nm)', size=14)
        X, Y = np.meshgrid(self.t1range, self.t2range)

        # heat map, no interpolation
        if not interp_contour:
            CS = axV2.pcolormesh(X, Y, J)
            axV2.set_xlim(self.t1range[0], self.t1range[-1])
            axV2.set_ylim(self.t2range[0], self.t2range[-1])
        else:  # contourf, interpolate data
            axV2.contourf(X, Y, J, 100, lw=0.1)
            CS = axV2.contourf(X, Y, J, 100)
        axV2.tick_params(labelsize=14)
        figV2.colorbar(CS)
        figV2.tight_layout()

        return None

    def CalS(self, L_vary=0):
        '''
        calculate S, S_prime, and S_dprime
        S = S' * L  * S"   for any j
             j    j    j

                        i = j-1
           S'[j]  = [  product  ( I      * L    )  ]   * I
          (j>0)         i = 0      i,i+1    i+1           j, j+1

        '''

        j = L_vary
        Imats, Lmats = self.Imats, self.Lmats
        nWL = len(self.WL)
        S_prime, S_dprime = self.S_prime, self.S_dprime

        layers = self.layers + ["Air"]

        # calculate S_prime and S

        if j in S_prime:
            S = np.copy(S_prime[j])
        else:
            S = np.array([np.eye(2, dtype=complex) for _ in range(nWL)])
            j = 0
        for matind in range(j + 1, len(layers)):
            pre, mater = layers[matind - 1], layers[matind]
            for i in range(nWL):
                S[i] = S[i].dot(Lmats[matind - 1][i])
                S[i] = S[i].dot(Imats[(pre, mater)][i])
            S_prime[matind] = np.copy(S)

        S_dprime[len(layers) - 2] = Imats[(layers[-2], layers[-1])]

        endind = len(layers) - 3 if j == 0 else j - 1
        # for matind in xrange(len(layers)-3, 0, -1):
        for matind in range(endind, 0, -1):
            mater, nex = layers[matind], layers[matind + 1]
            tmp = np.copy(S_dprime[matind + 1])
            for i in range(nWL):
                tmp[i] = np.dot(Lmats[matind + 1][i], tmp[i])
                tmp[i] = np.dot(Imats[(mater, nex)][i], tmp[i])
            S_dprime[matind] = tmp

        return S

# To do:
    def SaveVTData(SaveName):
        return


if __name__ == "__main__":

    demo = True
    if demo:
        Device = [("Glass", 0),
                  ("ITO", 145),
                  ("ZnO", 120),
                  ("PbS", 250),
                  ("Au", 150)]

        libname = "Index_of_Refraction_library_Demo.csv"
        WLrange = [350, 1200]
        ToVary = 3
        target = 3
        t_range = np.arange(50, 400, 20)
        VT = OMVaryThickness(Device, libname=libname, WLrange=WLrange)
        VT.VaryOne(ToVary, t_range, target, cbarlegend=True)
        plt.show()
