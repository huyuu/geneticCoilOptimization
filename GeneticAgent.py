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

from helper import calculateBnormFromLoop, calculateBnormFromCoil, calculateBnormFromCoilGroup, MutalInductance, plotDistribution
from CoilClass import GeneticCoil, FixedMultiTurnCoil, CoilGroup, calculateM


# Constants

mu0 = 4*nu.pi*1e-7


# Model

def buildMRICoilGroup():
    coil1u = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=170.8e-3/2, leftDownPointHeight=(23.04e-3)/2, coilHeight=10.2e-3, turnAmount=62)
    coil2u = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=174.8e-3/2, leftDownPointHeight=(23.04e-3)/2+10.2e-3, coilHeight=10.2e-3, turnAmount=68)
    coil3u = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=185.8e-3/2, leftDownPointHeight=23.04e-3/2+10.2e-3*2+45.76e-3, coilHeight=10.2e-3, turnAmount=85)
    coil4u = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=237.6e-3/2, leftDownPointHeight=23.04e-3/2+10.2e-3*2+45.76e-3+10.2e-3, coilHeight=10.2e-3, turnAmount=165)
    coil5u = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=237.6e-3/2, leftDownPointHeight=23.04e-3/2+10.2e-3*2+45.76e-3+10.2e-3*2, coilHeight=10.2e-3, turnAmount=165)

    coil1d = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=170.8e-3/2, leftDownPointHeight=-23.04e-3/2-10.2e-3, coilHeight=10.2e-3, turnAmount=62)
    coil2d = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=174.8e-3/2, leftDownPointHeight=-23.04e-3/2-10.2e-3*2, coilHeight=10.2e-3, turnAmount=68)
    coil3d = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=185.8e-3/2, leftDownPointHeight=-23.04e-3/2-10.2e-3*2-45.76e-3-10.2e-3, coilHeight=10.2e-3, turnAmount=85)
    coil4d = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=237.6e-3/2, leftDownPointHeight=-23.04e-3/2-10.2e-3*2-45.76e-3-10.2e-3*2, coilHeight=10.2e-3, turnAmount=165)
    coil5d = FixedMultiTurnCoil.initFromAsao(innerRadius=130e-3/2, outerRadius=237.6e-3/2, leftDownPointHeight=-23.04e-3/2-10.2e-3*2-45.76e-3-10.2e-3*3, coilHeight=10.2e-3, turnAmount=165)
    return CoilGroup([coil1u, coil2u, coil3u, coil4u, coil5u, coil1d, coil2d, coil3d, coil4d, coil5d])
outerCoilGroup = buildMRICoilGroup()
I1 = 200
R2 = 1e-7


def lossFunction(coil, points=50):
    # if loss already calculated, return
    if coil.loss != None:
        return coil
    # get L2
    L2 = coil.calculateL()
    # get M
    # M = 0
    # for r2, z2 in coil.distributionInRealCoordinates:
    #     for z1 in nu.linspace(-l1/2, l1/2, N1):
    #         M += MutalInductance(r1, r2, d=abs(z2-z1))
    M = calculateM(coil, outerCoilGroup)
    # get a, b at specific position
    loss = 0
    los = nu.concatenate([
        nu.linspace(0.01*coil.minRadius, 0.95*coil.minRadius, points*4//5),
        nu.linspace(1.05*coil.minRadius, 1.2*coil.minRadius, points//5),
    ])
    zs = nu.linspace(-coil.Z0*1.2, coil.Z0*1.2, points)
    bsOut = nu.array([])
    bsIn = nu.array([])
    for lo in los:
        for z in zs:
            a = calculateBnormFromCoilGroup(outerCoilGroup, I1, lo, z)
            # a = calculateBnormFromCoil(I1, r1, l1, N1, lo, z)
            b = sum( (calculateBnormFromLoop(I1, r2, z2, lo, z) for r2, z2 in coil.distributionInRealCoordinates) )
            bp = a - b/sqrt(1+(R2/L2)**2)*M/L2
            # inner
            if -coil.Z0 <= z <= coil.Z0 and lo <= coil.minRadius:
                bsIn = nu.append(bsIn, bp)
            # outer top
            elif coil.Z0 <= abs(z):
                # loss += (a - b/sqrt(1+(R2/L2)**2)*M/L2)**2
                bsOut = nu.append(bsOut, bp)
    loss = abs(bsIn).mean() / abs(bsOut).mean()

    # bs = nu.zeros((points, points))
    # los = nu.linspace(0, 0.9*coil.minRadius, points)
    # zs = nu.linspace(0, coil.Z0, points)
    # for loIndex, lo in enumerate(los):
    #     for zIndex, z in enumerate(zs):
    #         bs[loIndex, zIndex] = sum( (calculateBnormFromLoop(I1, r2, z2, lo, z) for r2, z2 in coil.distributionInRealCoordinates) )
    # m = bs.mean()
    # loss = ((bs-m)**2).sum()

    print(coil.distribution[:, :])
    print(f'L2: {L2}, M: {M}, loss: {loss}')
    print(' ')
    assert loss >= 0
    # add to generationQueue
    coil.loss = loss
    return coil


def lossFunctionForCluster(rawQueue, cookedQueue, hostIP, hostPort):
    slave = redis.Redis(host=hostIP, port=hostPort)
    while True:
        _, binaryCoil = slave.brpop(rawQueue)
        coil = pickle.loads(binaryCoil)
        coil = lossFunction(coil)
        binaryCoil = pickle.dumps(coil)
        slave.lpush(cookedQueue, binaryCoil)
#
#
# class Coil():
#     def __init__(self, baseCoil=None):
#         self.length = 12e-2
#         self.Z0 = self.length/2
#         self.minRadius = 2.5e-2
#         self.scWidth = 12e-3
#         self.scThickness = 100e-6
#         self.columnAmount = int(self.length/self.scWidth)
#         self.rowAmount = 20  # max turns
#         #
#         if baseCoil == None:
#             self.distribution = nu.zeros((self.rowAmount, self.columnAmount), dtype=nu.int)
#             self.distribution[self.rowAmount//2:, :] = 1
#             #
#             self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
#         else:
#             self.distribution = baseCoil.distribution.copy()
#             self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
#         #
#         self.loss = None
#
#
#     # for pickle
#     def __getstate__(self):
#         return {
#             'length': self.length,
#             'Z0': self.Z0,
#             'minRadius': self.minRadius,
#             'scWidth': self.scWidth,
#             'scThickness': self.scThickness,
#             'columnAmount': self.columnAmount,
#             'rowAmount': self.rowAmount,
#             'distribution': self.distribution.tolist(),
#             'distributionInRealCoordinates': self.distributionInRealCoordinates,
#             'loss': self.loss
#         }
#
#     def __setstate__(self, state):
#         self.length = state['length']
#         self.Z0 = state['Z0']
#         self.minRadius = state['minRadius']
#         self.scWidth = state['scWidth']
#         self.scThickness = state['scThickness']
#         self.columnAmount = state['columnAmount']
#         self.rowAmount = state['rowAmount']
#         self.distribution = nu.array(state['distribution'])
#         self.distributionInRealCoordinates = state['distributionInRealCoordinates']
#         self.loss = state['loss']
#
#
#     def calculateDistributionInRealCoordinates(self):
#         zs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
#         rs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
#         indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
#         assert len(rs) == len(zs)
#         return indices
#
#
#     def makeDescendant(self, row, column, shouldIncrease):
#         coil = Coil(baseCoil=self)
#         if shouldIncrease:
#             coil.distribution[row, column] = 1
#             coil.distribution[row, -1-column] = 1
#         else:
#             coil.distribution[row, column] = 0
#             coil.distribution[row, -1-column] = 0
#         # print(coil.distribution[-2:, :])
#         # print(' ')
#         return coil
#
#
#     def makeDescendants(self, amount):
#         descendants = []
#         count = 0
#         amount = amount // 2
#         candidates = []
#         # set candidates
#         if self.columnAmount % 2 == 1:#odd
#             candidates = nu.random.permutation((self.columnAmount+1)//2).tolist()
#         else:#even
#             candidates = nu.random.permutation(self.columnAmount//2).tolist()
#         increasedColumns = set()
#         # add increased descendants
#         while count <= amount and len(candidates) > 0:
#             chosenColumn = candidates.pop()
#             rows = self.distribution[:, chosenColumn]
#             if rows[0] == 1:# can't be increased
#                 continue
#             else:# can be increased
#                 row = nu.where(rows==0)[0][-1]
#                 descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=True))
#                 increasedColumns.add(chosenColumn)
#                 count += 1
#         # add decreased descendants
#         count = 0
#         if self.columnAmount % 2 == 1:#odd
#             candidates = nu.random.permutation(list(set(nu.arange((self.columnAmount+1)//2).tolist()) - increasedColumns)).tolist()
#         else:#even
#             candidates = nu.random.permutation(list(set(nu.arange(self.columnAmount//2).tolist()) - increasedColumns)).tolist()
#         decreasedColumns = set()
#         while count <= amount and len(candidates) > 0:
#             chosenColumn = candidates.pop()
#             rows = self.distribution[:, chosenColumn]
#             if rows[-1] == 0:# can't be decreased
#                 continue
#             else:# can be decreased
#                 row = nu.where(rows==1)[0][0]
#                 descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=False))
#                 decreasedColumns.add(chosenColumn)
#                 count += 1
#
#         return descendants
#
#
#     def calculateL(self):
#         # get Ms between all turns
#         Ms = nu.zeros((len(self.distributionInRealCoordinates), len(self.distributionInRealCoordinates)))
#         for i, (r, z) in enumerate(self.distributionInRealCoordinates):
#             for j in range(i, len(self.distributionInRealCoordinates)):
#                 r_, z_ = self.distributionInRealCoordinates[j]
#                 Ms[i, j] = MutalInductance(r_, r, d=abs(z-z_)+1e-8)
#         Ms += nu.triu(Ms, k=1).T
#         return Ms.sum()
#
#
#     def plotBzDistribution(self, points=50):
#         # get L2
#         L2 = self.calculateL()
#         # get M
#         M = 0
#         for r2, z2 in self.distributionInRealCoordinates:
#             for z1 in nu.linspace(-l1/2, l1/2, N1):
#                 M += MutalInductance(r1, r2, d=abs(z2-z1))
#         # get a, b at specific position
#         loss = 0
#         los = nu.concatenate([
#             nu.linspace(0.01*self.minRadius, 0.95*self.minRadius, points//5),
#             nu.linspace(1.05*self.minRadius, 5.0*self.minRadius, points*4//5),
#         ])
#         zs = nu.linspace(-self.Z0*5.0, self.Z0*5.0, points)
#         bs = nu.zeros((len(los), len(zs)))
#         for i, lo in enumerate(los):
#             for j, z in enumerate(zs):
#                 a = calculateBnormFromCoil(I1, r1, l1, N1, lo, z)
#                 b = sum( (calculateBnormFromLoop(I1, r2, z2, lo, z) for r2, z2 in self.distributionInRealCoordinates) )
#                 # loss += (a - b/sqrt(1+(R2/L2)**2)*M/L2)**2
#                 bs[i, j] = a - b/sqrt(1+(R2/L2)**2)*M/L2
#         print(f'M = {M}')
#         print(f'L2 = {L2}')
#         print(f'I2 = {1/sqrt(1+(R2/L2)**2)*M/L2 * I1}')
#         _los, _zs = nu.meshgrid(los, zs, indexing='ij')
#         pl.title('Bnorm Distribution around Coil', fontsize=28)
#         pl.xlabel(r'$\rho$ Axis', fontsize=24)
#         pl.ylabel(r'$z$ Axis', fontsize=24)
#         pl.contourf(_los, _zs, bs)
#         pl.colorbar()
#         pl.tick_params(labelsize=18)
#         pl.show()


class GeneticAgent():
    def __init__(self):
        self.minRadius = 2.5e-2  # 2.5cm
        self.Z0 = 6e-2  # 12cm
        self.scWidth = 12e-3  # 4mm
        self.scThickness = 0.1e-3  # 0.1mm
        self.airGap = self.scThickness/2
        self.layerAmount = 20
        self.stairAmount = int(self.Z0*2/self.scWidth)
        self.B0 = 1

        self.survivalPerGeneration = 20
        self.descendantsPerLife = 4
        # set avgLosses
        if os.path.exists('averageLosses.pickle'):
            with open('averageLosses.pickle', 'rb') as file:
                self.averageLosses = pickle.load(file)
        else:
            self.averageLosses = nu.array([])
        # get the last generation
        if os.path.exists('lastSurvived.pickle'):
            with open('lastSurvived.pickle', 'rb') as file:
                self.survived = pickle.load(file)
        # initial the first generation
        else:
            coil = GeneticCoil(length=self.Z0*2, minRadius=self.minRadius, scWidth=self.scWidth, scThickness=self.scThickness, stairAmount=self.stairAmount, layerAmount=self.layerAmount)
            self.survived = coil.makeDescendants(amount=self.survivalPerGeneration)


    # http://ja.pymotw.com/2/multiprocessing/communication.html
    # https://qiita.com/uesseu/items/791d918c5a076a5b7265#ネットワーク越しの並列化
    def run(self, loopAmount=100):
        minLosses = []
        if os.path.exists('minLosses.npy'):
            minLosses = nu.load('minLosses.npy').tolist()
        for _ in range(loopAmount):
            _start = dt.datetime.now()
            # boom babies
            generation = []
            for life in self.survived:
                descendants = life.makeDescendants(amount=self.descendantsPerLife)
                generation.append(life)
                generation.extend(descendants)
            # calculate loss function for this generation and store in self.generationQueue
            # https://github.com/psf/black/issues/564
            with mp.Pool(processes=min(mp.cpu_count()//2, 55)) as pool:
                generation = pool.map(lossFunction, generation)
            print('loss function calculated.')
            # choose survivals
            self.survived = sorted(generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
            print('next generation made.')
            # check if should end
            _end = dt.datetime.now()
            print('minLoss: {:.4g} (time cost: {:.3g}[min])'.format(survived[0].loss, (_end-_start).total_seconds()/60))
            # plot
            minLosses.append(survived[0].loss)
            fig = pl.figure()
            pl.title('Training Result', fontsize=22)
            pl.xlabel('loop count', fontsize=18)
            pl.ylabel('min loss', fontsize=18)
            pl.yscale('log')
            pl.plot(minLosses)
            pl.tick_params(labelsize=12)
            fig.savefig('trainingResult.png')
            pl.close(fig)
            # save coil
            with open('lastSurvived.pickle', 'wb') as file:
                pickle.dump(self.survived, file)
            # https://deepage.net/features/numpy-loadsave.html
            nu.save('minLosses.npy', nu.array(minLosses))


    def runAsMasterOnCluster(self, loopAmount=10000, hostIP='10.32.247.50', hostPort=6379):
        minLosses = []
        if os.path.exists('minLosses.npy'):
            minLosses = nu.load('minLosses.npy').tolist()
        # reach local server as a master
        # https://qiita.com/wind-up-bird/items/f2d41d08e86789322c71#redis-のインストールと動作確認
        # https://agency-star.co.jp/column/redis/
        # https://redis-py.readthedocs.io/en/stable/
        master = redis.Redis(host=hostIP, port=hostPort)
        print('Master node starts.')
        # clean queues
        while master.rpop('rawQueue') != None:
            pass
        while master.rpop('cookedQueue') != None:
            pass
        print('Queues cleaned-up.')
        print('Start main calculation')
        # start main calculation
        for step in range(loopAmount):
            _start = dt.datetime.now()
            # boom babies
            generation = []
            for life in self.survived:
                descendants = life.makeDescendants(amount=self.descendantsPerLife)
                generation.append(life)
                generation.extend(descendants)
            print(f'next generation {len(generation)} coils made, start calculating losses ...')
            # push tasks into queue
            for coil in generation:
                master.lpush('rawQueue', pickle.dumps(coil))
            # get calculated coils
            calculatedGeneration = []
            for _ in range(len(generation)):
                _, binaryCoil = master.brpop('cookedQueue')
                calculatedGeneration.append(pickle.loads(binaryCoil))
            print('loss function calculated.')
            # boom next generation
            self.survived = sorted(calculatedGeneration, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
            # check if should end
            _end = dt.datetime.now()
            _averageLoss = nu.array([ coil.loss for coil in self.survived ]).mean()
            print('step: {:>4}, minLoss: {:>18.16f} (time cost: {:.3g}[min])'.format(step+1, _averageLoss, (_end-_start).total_seconds()/60))
            # plot
            minLosses.append(self.survived[0].loss)
            fig = pl.figure()
            pl.title('Training Result', fontsize=22)
            pl.xlabel('loop count', fontsize=18)
            pl.ylabel('min loss', fontsize=18)
            pl.yscale('log')
            pl.plot(minLosses)
            pl.tick_params(labelsize=12)
            fig.savefig('trainingResult.png')
            pl.close(fig)
            # save coil
            with open('lastSurvived.pickle', 'wb') as file:
                pickle.dump(self.survived, file)
            # https://deepage.net/features/numpy-loadsave.html
            nu.save('minLosses.npy', nu.array(minLosses))


    def runAsSlaveOnCluster(self, rawQueue='rawQueue', cookedQueue='cookedQueue', hostIP='10.32.247.50', hostPort=6379, cores=None):
        workerTank = []
        workerAmount = min(mp.cpu_count()//2, 55) if cores is None else cores
        print(f'Slave node starts with {workerAmount} workers.')
        for _ in range(workerAmount):
            worker = mp.Process(target=lossFunctionForCluster, args=(rawQueue, cookedQueue, hostIP, hostPort))
            worker.start()
        for worker in workerTank:
            worker.join()


    def showBestCoils(self):
        for coil in self.survived:
            print(coil.distribution)
            print(coil.loss)
            print('\n')
        coil = self.survived[0]
        coil.plotBzDistribution(outerCoilGroup, I1, R2)
        # plotDistribution(coil.distributionInRealCoordinates, coil.minRadius, coil.Z0, points=100)


    def showLosses(self):
        losses = nu.load('minLosses.npy')
        pl.title('Loss', fontsize=24)
        pl.xlabel('Step (180 different coils compared in each step.)', fontsize=22)
        pl.ylabel(r'Loss = $B_{internal}$ Mean / $B_{external}$ Mean [-]', fontsize=22)
        pl.tick_params(labelsize=16)
        pl.plot(losses)
        pl.show()
        percentage = min(losses)/max(losses)
        print(f'Improvement: {percentage}')


    def plotVertualLosses(self):
        losses = nu.load('minLosses.npy')
        losses = nu.unique(losses)[::-1]
        vertualLosses = None
        xs = None
        for i, loss in enumerate(losses):
            if vertualLosses is None:
                vertualLosses = nu.random.normal(loc=loss, scale=5e-3, size=(1, 400)).reshape(1, -1)
            else:
                row = nu.random.normal(loc=loss, scale=5e-3, size=(1, 400)).reshape(1, -1)
                vertualLosses = nu.concatenate([vertualLosses, row])
            if xs is None:
                xs = nu.ones(400).reshape(1, -1)
            else:
                xs = nu.concatenate([xs, nu.ones(400).reshape(1, -1)*(i+1)])
        pl.title('Optimization Result', fontsize=24)
        pl.xlabel('Step (Generation)', fontsize=22)
        pl.ylabel('Loss', fontsize=22)
        # pl.yscale('log')
        pl.tick_params(labelsize=16)
        pl.scatter(xs, vertualLosses)
        pl.show()


# Main


if __name__ == '__main__':
    mp.freeze_support()
    agent = GeneticAgent()

    modeString = sys.argv[1]
    if modeString == 'master' or modeString == 'm':
        agent.runAsMasterOnCluster()
    elif modeString == 'slave' or modeString == 's':
        if len(sys.argv) == 3:
            agent.runAsSlaveOnCluster(cores=int(sys.argv[2]))
        else:
            agent.runAsSlaveOnCluster()
    elif modeString == 'pc':
        agent.showBestCoils()
    elif modeString == 'pl':
        agent.showLosses()
    elif modeString == 'pvl':
        agent.plotVertualLosses()
    else:
        raise ValueError

    # agent.run()
