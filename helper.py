import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
from numpy import abs, sqrt, cos, sin, pi

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7
I = 100


# Model

# phi_: coil phi, a: coil radius, z_: coil z position, lo: point p's radius, z: point p's z position
def _lo(phi_, a, z_, lo, z):
    return (z-z_)*cos(phi_) / (lo**2 + a**2 - 2*lo*a*cos(phi_) + (z-z_)**2)**1.5

# phi_: coil phi, a: coil radius, z_: coil z position, lo: point p's radius, z: point p's z position
def _z(phi_, a, z_, lo, z):
    return (a-lo*cos(phi_)) / (lo**2 + a**2 - 2*lo*a*cos(phi_) + (z-z_)**2)**1.5

# http://www.f-denshi.com/000TokiwaJPN/20vectr/cpx01.html
def BpFromBiosavart(I, coilRadius, coilZ, lo, z):
    Bp_r = mu0*I/4/pi*coilRadius * quadrature(_lo, 0, 2*pi, args=(coilRadius, coilZ, lo, z), maxiter=10000)[0]
    Bp_z = mu0*I/4/pi*coilRadius * quadrature(_z, 0, 2*pi, args=(coilRadius, coilZ, lo, z), maxiter=10000)[0]
    return (Bp_r, Bp_z)


def BpFromVectorPotential(I, coilRadius, coilZ, lo, z):
    squaredK = 4*coilRadius*lo/( (coilRadius+lo)**2 + (z-coilZ)**2 )
    k = sqrt(squaredK)
    Aphi = lambda z: mu0*I/pi * ( (sqrt((coilRadius+lo)**2+(z-coilZ)**2)/(2*lo) - coilRadius/sqrt((coilRadius+lo)**2+(z-coilZ)**2))*ellipk(squaredK) - ellipe(squaredK) )
    dAphi_dz = (Aphi(z*1.0001) - Aphi(z))/abs(z*0.0001)

    loAphi = lambda lo: mu0*I/pi * ( (0.5*sqrt((coilRadius+lo)**2+(z-coilZ)**2) - coilRadius*lo/sqrt((coilRadius+lo)**2+(z-coilZ)**2))*ellipk(squaredK) - ellipe(squaredK) )
    dloAphi_dlo = (loAphi(lo*1.0001) - loAphi(lo))/abs(lo*0.0001)

    Bp_r = - dAphi_dz
    Bp_z = 1/lo*dloAphi_dlo
    return (Bp_r, Bp_z)

    # dk_dlo = 0.5/k * ( 4*coilRadius/((coilRadius+lo)**2+(z-coilZ)**2) - 4*coilRadius*lo/((coilRadius+lo)**2+(z-coilZ)**2)**2 * 2*(coilRadius+coilZ) ) if lo != 0 else 0
    # dk_dz = -4*coilRadius*lo/(2*k) * ((coilRadius+lo)**2+(z-coilZ)**2)**(-2) * 2*(z-coilZ) if lo != 0 else 0
    # if k >= 0.9:
    #     Bp_r = -mu0*I/pi * 1/k * sqrt(coilRadius/lo) * ( -1/k**2 * ellipkm1(squaredK) + (1/k-k/2)/(k*(1-k**2))*ellipe(squaredK) ) * dk_dz
    #     Bp_z = mu0*I/pi * sqrt(coilRadius/lo) * ( ((0.5/k-k/4)/lo+(1-1/k**2)) * ellipkm1(squaredK) + (-1/(2*k*lo)+(1/k-k/2)/(k*(1-k**2))) * ellipe(squaredK) ) * dk_dlo
    # else:
    #     Bp_r = -mu0*I/pi * 1/k * sqrt(coilRadius/lo) * ( -1/k**2 * ellipk(squaredK) + (1/k-k/2)/(k*(1-k**2))*ellipe(squaredK) ) * dk_dz
    #     Bp_z = mu0*I/pi * sqrt(coilRadius/lo) * ( ((0.5/k-k/4)/lo+(1-1/k**2)) * ellipk(squaredK) + (-1/(2*k*lo)+(1/k-k/2)/(k*(1-k**2))) * ellipe(squaredK) ) * dk_dlo
    # return (Bp_r, Bp_z)


def _Bmag_lo(z_, lo, z, lom):
    lo_ = lom(z_)
    beta = (lo_+lo)**2 + (z-z_)**2
    beta_r = sqrt(beta)
    squaredK = 4*lo*lo_ / beta
    k = sqrt(squaredK)
    dk_dz = -sqrt(4*lo*lo_) * beta**(-1.5) * (z-z_)
    return ( (z-z_)/beta_r + 2*lo_*lo*(z-z_)/beta**1.5 )*ellipk(squaredK) +\
    ( beta_r - 2*lo*lo_/beta_r )*ellipk_dk(squaredK)*dk_dz -\
    ( (z-z_)/beta_r )*ellipe(squaredK) -\
    ( beta_r )*ellipe_dk(squaredK)*dk_dz


def _Bmag_z(z_, lo, z, lom):
    lo_ = lom(z_)
    beta = (lo_+lo)**2 + (z-z_)**2
    beta_r = sqrt(beta)
    squaredK = 4*lo*lo_ / beta
    k = sqrt(squaredK)
    dk_dlo = sqrt(lo_/lo/beta) - 2*sqrt(lo_*lo)*(lo+lo_)*beta**(-1.5)
    return ( (lo_+lo)/beta_r - 2*lo_/beta_r + 2*lo*lo_*(lo+lo_)/beta**1.5 )*ellipk(squaredK) +\
    ( beta_r - 2*lo*lo_/beta_r )*ellipk_dk(squaredK)*dk_dlo -\
    ( (lo+lo_)/beta_r )*ellipe(squaredK) -\
    ( beta_r )*ellipe_dk(squaredK)*dk_dlo


def BzCoil(lo, z, lo_, coilZs, I):
    # Bs_lo = 0
    Bs_z = 0
    lom = lambda x: lo_
    for i, z_ in enumerate(coilZs):
        # Bs_lo += -1.0 * _Bmag_lo(z_, lo, z, lom)
        Bs_z += 1.0 * _Bmag_z(z_, lo, z, lom)
    # return mu0*I/(2*pi*lo) * nu.array([Bs_lo, Bs_z])
    return mu0*I/(2*pi*lo) * Bs_z


def calculateBzFromLoop(I, coilRadius, coilZ, lo, z):
    return BzCoil(lo=lo, z=z, lo_=coilRadius, coilZs=[coilZ], I=I)


def calculateBzFromCoil(I, r, l, N, lo, z):
    coilZPositions = nu.linspace(-l/2, l/2, N)
    return sum((calculateBzFromLoop(I, r, coilZ, lo, z) for coilZ in coilZPositions))


# def calculateBFromCoil(coilCoordinates, minRadius, Z0, lo, z, points):
#     bp_r = 0
#     bp_z = 0
#     for a, z_ in coilCoordinates:
#         bp = BpFromBiosavart(I1, a, z_, lo, z)
#         bp_r += bp[0]
#         bp_z += bp[1]
#     return (bp_r, bp_z)


def plotDistribution(coilCoordinates, minRadius, Z0, points):
    los = nu.linspace(0, 0.9*minRadius, points)
    zs = nu.linspace(0, Z0, points)
    # create args
    args = []
    for lo in los:
        for z in zs:
            args.append((I, coilRadius, coilZ, lo, z))
    # calculate bs for all points
    bs = []
    with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
        bs = pool.starmap(calculateBFromCoil, args)
    bs_r = nu.array([ b[0] for b in bs ]).reshape((points, points))
    bs_z = nu.array([ b[1] for b in bs ]).reshape((points, points))
    # plot
    pl.xlabel(r'$\rho$/coil_radius')
    pl.ylabel(r'$Z-Z_0$')
    X, Y = nu.meshgrid(los/coilRadius, zs-coilZ, indexing='ij')
    pl.quiver(X, Y, bs_r, bs_z)
    pl.show()



def _f(phi, r1, r2, d):
    return r1 * r2 * nu.cos(phi) / nu.sqrt( r1**2 + r2**2 + d**2 - 2*r1*r2*nu.cos(phi) )

def MutalInductance(r1, r2, d):
    # return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-6, maxiter=100000)[0]
    if r1 == 0:
        r1 += 1e-8
    if r2 == 0:
        r2 += 1e-8
    squaredK = 4*r1*r2/((r1+r2)**2+d**2)
    k = nu.sqrt(squaredK) if squaredK != 0 else 0
    if k < 0.9:
        result = mu0 * nu.sqrt(r1*r2) * ( (2/k-k)*ellipk(squaredK) - 2/k*ellipe(squaredK) )
    else:  # k around 1
        result = mu0 * nu.sqrt(r1*r2) * ( (2/k-k)*ellipkm1(squaredK) - 2/k*ellipe(squaredK) )

    if result >= 0:
        return result
    else:
        return 0.5 * mu0 * quadrature(_f, 0, 2*nu.pi, args=(r1, r2, d), tol=1e-6, maxiter=10000)[0]




if __name__ == '__main__':
    coilRadius = 1.5e-2
    coilZ = 0
    points = 50
    Z0 = coilRadius
    los = nu.linspace(0.1*coilRadius, 0.9*coilRadius, points)
    zs = nu.linspace(0, Z0, points)

    # create args
    args = []
    for lo in los:
        for z in zs:
            args.append((I, coilRadius, coilZ, lo, z))
    # calculate bs for all points
    bs = []
    with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
        bs = pool.starmap(BpFromBiosavart, args)
    bs_r = nu.array([ b[0] for b in bs ]).reshape((points, points))
    bs_z = nu.array([ b[1] for b in bs ]).reshape((points, points))
    pl.xlabel(r'$\rho$/coil_radius')
    pl.ylabel(r'$Z-Z_0$')
    X, Y = nu.meshgrid(los/coilRadius, zs-coilZ, indexing='ij')
    pl.quiver(X, Y, bs_r, bs_z)
    pl.show()
