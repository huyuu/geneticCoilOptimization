import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.integrate import quadrature
from numpy import cos, sin, pi, abs, sqrt
import multiprocessing as mp
import math as ma
import datetime as dt
import os
import pickle
import redis
import sys
from helper import calculateBnormFromLoop, calculateBnormFromCoil, MutalInductance, plotDistribution


class GeneticCoil():
    def __init__(self, length, minRadius, scWidth, scThickness, stairAmount, layerAmount):
        self.length = length
        self.Z0 = self.length/2
        self.minRadius = minRadius
        self.scWidth = scWidth
        self.scThickness = scThickness
        self.columnAmount = stairAmount
        self.rowAmount = layerAmount  # max turns

        self.distribution = nu.zeros((self.rowAmount, self.columnAmount), dtype=nu.int)
        self.distribution[self.rowAmount//2:, :] = 1
        self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        self.loss = None


    @classmethod
    def initFromBaseCoil(cls, baseCoil):
        coil = cls(length=baseCoil.length, minRadius=baseCoil.minRadius, scWidth=baseCoil.scWidth, scThickness=baseCoil.scThickness, stairAmount=baseCoil.columnAmount, layerAmount=baseCoil.rowAmount)
        coil.distribution = baseCoil.distribution.copy()
        coil.distributionInRealCoordinates = baseCoil.distributionInRealCoordinates.copy()
        coil.loss = baseCoil.loss
        return coil


    # for pickle
    def __getstate__(self):
        return {
            'length': self.length,
            'Z0': self.Z0,
            'minRadius': self.minRadius,
            'scWidth': self.scWidth,
            'scThickness': self.scThickness,
            'columnAmount': self.columnAmount,
            'rowAmount': self.rowAmount,
            'distribution': self.distribution.tolist(),
            'distributionInRealCoordinates': self.distributionInRealCoordinates,
            'loss': self.loss
        }

    def __setstate__(self, state):
        self.length = state['length']
        self.Z0 = state['Z0']
        self.minRadius = state['minRadius']
        self.scWidth = state['scWidth']
        self.scThickness = state['scThickness']
        self.columnAmount = state['columnAmount']
        self.rowAmount = state['rowAmount']
        self.distribution = nu.array(state['distribution'])
        self.distributionInRealCoordinates = state['distributionInRealCoordinates']
        self.loss = state['loss']


    def calculateDistributionInRealCoordinates(self):
        zs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
        rs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
        indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
        assert len(rs) == len(zs)
        return indices


    def makeDescendant(self, row, column, shouldIncrease):
        coil = GeneticCoil.initFromBaseCoil(baseCoil=self)
        if shouldIncrease:
            coil.distribution[row, column] = 1
            coil.distribution[row, -1-column] = 1
        else:
            coil.distribution[row, column] = 0
            coil.distribution[row, -1-column] = 0
        # print(coil.distribution[-2:, :])
        # print(' ')
        return coil


    def makeDescendants(self, amount):
        descendants = []
        count = 0
        amount = amount // 2
        candidates = []
        # set candidates
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation((self.columnAmount+1)//2).tolist()
        else:#even
            candidates = nu.random.permutation(self.columnAmount//2).tolist()
        increasedColumns = set()
        # add increased descendants
        while count <= amount and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[0] == 1:# can't be increased
                continue
            else:# can be increased
                row = nu.where(rows==0)[0][-1]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=True))
                increasedColumns.add(chosenColumn)
                count += 1
        # add decreased descendants
        count = 0
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation(list(set(nu.arange((self.columnAmount+1)//2).tolist()) - increasedColumns)).tolist()
        else:#even
            candidates = nu.random.permutation(list(set(nu.arange(self.columnAmount//2).tolist()) - increasedColumns)).tolist()
        decreasedColumns = set()
        while count <= amount and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[-1] == 0:# can't be decreased
                continue
            else:# can be decreased
                row = nu.where(rows==1)[0][0]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=False))
                decreasedColumns.add(chosenColumn)
                count += 1

        return descendants


    def calculateL(self):
        # get Ms between all turns
        Ms = nu.zeros((len(self.distributionInRealCoordinates), len(self.distributionInRealCoordinates)))
        for i, (r, z) in enumerate(self.distributionInRealCoordinates):
            for j in range(i, len(self.distributionInRealCoordinates)):
                r_, z_ = self.distributionInRealCoordinates[j]
                Ms[i, j] = MutalInductance(r_, r, d=abs(z-z_)+1e-8)
        Ms += nu.triu(Ms, k=1).T
        return Ms.sum()


    def plotBzDistribution(self, points=50):
        # get L2
        L2 = self.calculateL()
        # get M
        M = 0
        for r2, z2 in self.distributionInRealCoordinates:
            for z1 in nu.linspace(-l1/2, l1/2, N1):
                M += MutalInductance(r1, r2, d=abs(z2-z1))
        # get a, b at specific position
        loss = 0
        los = nu.concatenate([
            nu.linspace(0.01*self.minRadius, 0.95*self.minRadius, points//5),
            nu.linspace(1.05*self.minRadius, 5.0*self.minRadius, points*4//5),
        ])
        zs = nu.linspace(-self.Z0*5.0, self.Z0*5.0, points)
        bs = nu.zeros((len(los), len(zs)))
        for i, lo in enumerate(los):
            for j, z in enumerate(zs):
                a = calculateBnormFromCoil(I1, r1, l1, N1, lo, z)
                b = sum( (calculateBnormFromLoop(I1, r2, z2, lo, z) for r2, z2 in self.distributionInRealCoordinates) )
                # loss += (a - b/sqrt(1+(R2/L2)**2)*M/L2)**2
                bs[i, j] = a - b/sqrt(1+(R2/L2)**2)*M/L2
        _los, _zs = nu.meshgrid(los, zs, indexing='ij')
        pl.contourf(_los, _zs, bs)
        pl.colorbar()
        pl.show()



class FixedMultiTurnCoil():
    def __init__(self, centerRadius, centerHeight, coilWidth, coilHeight, turnAmount):
        self.centerRadius = centerRadius
        self.centerHeight = centerHeight
        self.coilWidth = coilWidth
        self.coilHeight = coilHeight
        self.turnAmount = turnAmount

        self.distributionInRealCoordinates = self.__calculateCoordinates()


    @classmethod
    def initFromAsao(cls, innerRadius, outerRadius, leftDownPointHeight, coilHeight, turnAmount):
        return cls(centerRadius=(innerRadius+outerRadius)/2, centerHeight=leftDownPointHeight+coilHeight/2, coilWidth=outerRadius-innerRadius, coilHeight=coilHeight, turnAmount=turnAmount)


    def __calculateCoordinates(self):
        rs = nu.linspace(-self.coilWidth/2, self.coilWidth/2, self.turnAmount) + self.centerRadius
        zs = nu.ones(self.turnAmount) * (self.centerHeight-self.coilHeight/2)
        assert len(rs) == len(zs)
        return [ (r, z) for r, z in zip(rs, zs) ]


class CoilGroup():
    def __init__(self, coils):
        self.coils = coils

        distribution = []
        for coil in coils:
            distribution.extend(coil.distributionInRealCoordinates)
        self.distributionInRealCoordinates = distribution



def calculateM(coilIn, coilOut):
    M = 0
    for rIn, zIn in coilIn.distributionInRealCoordinates:
        for rOut, zOut in coilOut.distributionInRealCoordinates:
            M += MutalInductance(rOut, rIn, d=abs(zIn-zOut))
    return M
