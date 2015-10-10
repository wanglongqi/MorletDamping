# -*- coding: utf-8 -*-
"""
This is a Python implementation of damping identification
method using Morlet wave, based on [1]:

[1] J. Slavič, M. Boltežar, Damping identification with the Morlet-
wave, Mechanical Systems and Signal Processing. 25 (2011) 1632–1645.
doi:10.1016/j.ymssp.2011.01.008. 

Check http://lab.fs.uni-lj.si/ladisk/?what=abstract&ID=58 for more info.

Created on Thu Oct 08 15:58:13 2015

@author: WANG Longqi
"""
from __future__ import print_function
import numpy as np
from scipy.integrate import simps, romb
from scipy.optimize import newton
from scipy.special import erf

class MorletDamping(object):
    _integration = simps
    _root_finding = "close"

    def __init__(self, sig, fs, k, n1=10, n2=20):
        """
        :param sig: analysed signal
        :param fs:  frequency of sampling
        :param k: number of oscillations for the damping identification
        :param n1: time-spread parameter
        :param n2: time-spread parameter
        :return:
        """
        self.sig = sig
        self.fs = float(fs)
        self.k = float(k)
        self.n1 = float(n1)
        self.n2 = float(n2)

    def identify_damping(self, w):
        """
        Identify damping at circular frequency `w` (rad/s)

        """
        M = np.abs(self.morlet_integrate(self.n1, w)) /\
            np.abs(self.morlet_integrate(self.n2, w))
        if self._root_finding == "close":
            return self.n1 * self.n2 / 2 / np.pi /\
                np.sqrt(self.k * self.k * (self.n2 *\
                self.n2 - self.n1 * self.n1)) *\
                np.sqrt(np.log(np.sqrt(self.n1 / self.n2) * M))
        else:
            # eq (19):
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
        """
        Perform the numerical integration with a Morlet wave at circular freq `w` and time-spread parameter `n`.

        :param n: time-spread parameter
        :param w: circular frequency (rad/s)
        :return:
        """
        eta = 2 * np.sqrt(2) * np.pi * self.k / n # eq (14)
        s = eta / w
        T = self.k * 2 * np.pi / w # eq (12)
        if T > (self.sig.size / self.fs):
            raise ValueError("Signal is too short, %d points are needed" % np.round(T * self.fs))
        npoints = np.round(T * self.fs)
        t = np.arange(npoints) / self.fs
        # From now on `t` is `t - T/2`
        t -= T/2
        kernel = np.exp(-t * t / s / s / 2) *\
            np.exp(-1j * eta * t / s)
        kernel *= 1 / (np.pi ** 0.25 * np.sqrt(s))
        return simps(self.sig[:npoints] * kernel, dx=1 / float(self.fs)) # eq (15)

if __name__ == "__main__":
    fs1 = 100
    t1 = np.arange(0, 6, 1. / fs1)
    w1 = 2 * np.pi * 10
    sig1 = np.cos(w1 * t1) * np.exp(-0.02 * w1 * t1)
    k1 = 40

#    Close form
    identifier = MorletDamping(sig1, fs1, k1, 10, 20)
    print(identifier.identify_damping(w1))
#    Exact
    identifier = MorletDamping(sig1, fs1, k1, 5, 10)
    identifier.set_root_finding(newton, 0.1)
    print(identifier.identify_damping(w1))
