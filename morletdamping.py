# -*- coding: utf-8 -*-
"""
This is a Python implementation of damping identification
method using Morlet wave, based on [1]:

[1] J. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing. 25 (2011) 1632–1645. doi:10.1016/j.ymssp.2011.01.008.

Created on Thu Oct 08 15:58:13 2015

@author: WANG Longqi
"""
import numpy as np
from scipy.integrate import simps, romb
from scipy.optimize import newton
from scipy.special import erf


class MorletDamping(object):
    _integration = simps
    _root_finding = "close"

    def __init__(self, sig, fs, k, n1=10, n2=20):
        self.sig = sig
        self.fs = float(fs)
        self.k = float(k)
        self.n1 = float(n1)
        self.n2 = float(n2)

    def identify_damping(self, w):
        M = np.abs(self.morlet_integrate(self.n1, w)) /\
            np.abs(self.morlet_integrate(self.n2, w))
        if self._root_finding == "close":
            return self.n1 * self.n2 / 2 / np.pi /\
                np.sqrt(self.k * self.k * (self.n2 *\
                self.n2 - self.n1 * self.n1)) *\
                np.sqrt(np.log(np.sqrt(self.n1 / self.n2) * M))
        else:
            eqn = lambda x: np.exp(4 * np.pi**2 * self.k**2\
                * x**2 * (self.n2**2 -\
                self.n1**2) / self.n1**2 / self.n2**2 ) *\
                np.sqrt(self.n2 / self.n1) *\
                (erf(2 * np.pi * self.k * x / self.n1 +\
                            self.n1 / 4)\
                    -erf(2 * np.pi * self.k * x / self.n1 -\
                            self.n1 / 4)) /\
                (erf(2 * np.pi * self.k * x / self.n2 +\
                            self.n2 / 4)\
                    -erf(2 * np.pi * self.k * x / self.n2 -\
                            self.n2 / 4)) - M
            return newton(eqn, self.x0)

    def set_int_method(self, method):
        self._integration = method

    def set_root_finding(self, method, x0=0):
        self._root_finding = method
        self.x0 = x0

    def morlet_integrate(self, n, w):
        eta = 2 * np.sqrt(2) * np.pi * self.k / n
        s = eta / w
        T = self.k * 2 * np.pi / w
        if T > (self.sig.size / self.fs):
            raise ValueError("Signal is too short, %d points are needed" % np.round(T * self.fs))
        npoints = np.round(T * self.fs)
        t = np.arange(npoints) / self.fs
        # From now on `t` is `t - T/2`
        t -= T/2
        kernel = np.exp(-t * t / s / s / 2) *\
            np.exp(-1j * eta * t / s)
        kernel *= 1 / (np.pi ** 0.25 * np.sqrt(s))
        return simps(self.sig[:npoints] * kernel, dx=1 / float(self.fs))

if __name__ == "__main__":
    fs1 = 100
    t1 = np.arange(0, 6, 1. / fs1)
    w1 = 2 * np.pi * 10
    sig1 = np.cos(w1 * t1) * np.exp(- 0.07 * w1 * t1)
    k1 = 40

#    Close form
    identifier = MorletDamping(sig1, fs1, k1, 10, 20)
    print identifier.identify_damping(w1)
#    Exact
    identifier = MorletDamping(sig1, fs1, k1, 5, 10)
    identifier.set_root_finding(newton, 0.1)
    print identifier.identify_damping(w1)
