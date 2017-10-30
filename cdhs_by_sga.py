#!/usr/bin/python

from exoplanets import exoplanets
from sga import gradient_ascent

exoplanets.dropna(how='any', inplace=True)


def construct_cd_and_derivative(alpha, beta, gamma, delta):
    def cd_hpf(R, D, Ts, Ve):
        return (R**alpha) * (D**beta) * (Ts**gamma) * (Ve**delta)
    def cd_hpf_der(R, D, Ts, Ve):
        return 
    return cd_hpf


# Estimating CD-HPF using gradient_descent.
