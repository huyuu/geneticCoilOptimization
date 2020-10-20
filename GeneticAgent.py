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

from helper import calculateBzFromLoop, calculateBzFromCoil, MutalInductance, plotDistribution


# Constants

mu0 = 4*nu.pi*1e-7


# Model


def lossFunction(coil, points=50):
    # if loss already calculated, return
    if coil.loss != None:
        return coil
    # get L2
    L2 = coil.calculateL()
    # get M
    M = 0
    for r2, z2 in coil.distributionInRealCoordinates:
        for z1 in nu.linspace(-l1/2, l1/2, N1):
            M += MutalInductance(r1, r2, d=abs(z2-z1)+1e-8)
    # get a, b at specific position
    loss = 0
    los = nu.concatenate([
        nu.linspace(0.01*coil.minRadius, 0.95*coil.minRadius, points//2),
        nu.linspace(1.05*coil.minRadius, 1.4*coil.minRadius, points//2),
    ])
    zs = nu.linspace(-coil.Z0*1.4, coil.Z0*1.4, points)
    for lo in los:
        for z in zs:
            if -coil.Z0*1.4 <= z <= coil.Z0*1.4 and lo <= coil.minRadius:
                continue
            a = calculateBzFromCoil(I1, r1, l1, N1, lo, z)
            b = sum( (calculateBzFromLoop(I1, r2, z2, lo, z) for r2, z2 in coil.distributionInRealCoordinates) )
            loss += (a - b/sqrt(1+(R2/L2)**2)*M/L2)**2

    # bs = nu.zeros((points, points))
    # los = nu.linspace(0, 0.9*coil.minRadius, points)
    # zs = nu.linspace(0, coil.Z0, points)
    # for loIndex, lo in enumerate(los):
    #     for zIndex, z in enumerate(zs):
    #         bs[loIndex, zIndex] = sum( (calculateBnormFromLoop(I1, r2, z2, lo, z) for r2, z2 in coil.distributionInRealCoordinates) )
    # m = bs.mean()
    # loss = ((bs-m)**2).sum()

    print(coil.distribution[-2:, :])
    print(f'L2: {L2}, M: {M}, loss: {loss}')
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


class Coil():
    def __init__(self, baseCoil=None):
        self.length = 10e-2
        self.Z0 = self.length/2
        self.minRadius = 1.5e-2
        self.scWidth = 4e-3
        self.scThickness = 100e-6
        self.columnAmount = int(self.length/self.scWidth)
        self.rowAmount = 4  # max turns
        #
        if baseCoil == None:
            self.distribution = nu.zeros((self.rowAmount, self.columnAmount), dtype=nu.int)
            self.distribution[[-1, -2], :] = 1
            #
            self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        else:
            self.distribution = baseCoil.distribution.copy()
            self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        #
        self.loss = None


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
        rs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
        zs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
        indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
        assert len(rs) == len(zs)
        return indices


    def makeDescendant(self, row, column, shouldIncrease):
        coil = Coil(baseCoil=self)
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


class GeneticAgent():
    def __init__(self):
        self.generation = []
        self.survivalPerGeneration = 20
        self.descendantsPerLife = 8
        # init generation
        if os.path.exists('lastGeneration.pickle'):
            with open('lastGeneration.pickle', 'rb') as file:
                self.generation = pickle.load(file)
        else:
            coil = Coil()
            for _ in range(self.survivalPerGeneration):
                self.generation.append(coil)


    # http://ja.pymotw.com/2/multiprocessing/communication.html
    # https://qiita.com/uesseu/items/791d918c5a076a5b7265#ネットワーク越しの並列化
    def run(self, loopAmount=100):
        minLosses = []
        if os.path.exists('minLosses.npy'):
            minLosses = nu.load('minLosses.npy').tolist()
        for _ in range(loopAmount):
            _start = dt.datetime.now()
            # calculate loss function for this generation and store in self.generationQueue
            # https://github.com/psf/black/issues/564
            with mp.Pool(processes=min(mp.cpu_count()-1, 55)) as pool:
                self.generation = pool.map(lossFunction, self.generation)
            print('loss function calculated.')
            # boom next generation
            survived = sorted(self.generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
            self.generation = []
            for life in survived:
                descendants = life.makeDescendants(amount=self.descendantsPerLife)
                self.generation.append(life)
                self.generation.extend(descendants)
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
            with open('lastGeneration.pickle', 'wb') as file:
                pickle.dump(survived, file)
            # https://deepage.net/features/numpy-loadsave.html
            nu.save('minLosses.npy', nu.array(minLosses))


    def runAsMasterOnCluster(self, loopAmount=100, hostIP='10.32.247.50', hostPort=6379):
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
        for _ in range(loopAmount):
            _start = dt.datetime.now()
            # push tasks into queue
            for coil in self.generation:
                master.lpush('rawQueue', pickle.dumps(coil))
            # get calculated coils
            calculatedGeneration = []
            for _ in range(len(self.generation)):
                _, binaryCoil = master.brpop('cookedQueue')
                calculatedGeneration.append(pickle.loads(binaryCoil))
            self.generation = calculatedGeneration
            print('loss function calculated.')
            # boom next generation
            survived = sorted(self.generation, key=lambda coil: coil.loss)[:self.survivalPerGeneration]
            self.generation = []
            for life in survived:
                descendants = life.makeDescendants(amount=self.descendantsPerLife)
                self.generation.append(life)
                self.generation.extend(descendants)
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
            with open('lastGeneration.pickle', 'wb') as file:
                pickle.dump(survived, file)
            # https://deepage.net/features/numpy-loadsave.html
            nu.save('minLosses.npy', nu.array(minLosses))


    def runAsSlaveOnCluster(self, rawQueue='rawQueue', cookedQueue='cookedQueue', hostIP='10.32.247.50', hostPort=6379):
        workerTank = []
        workerAmount = min(mp.cpu_count()-1, 55)
        print(f'Slave node starts with {workerAmount} workers.')
        for _ in range(workerAmount):
            worker = mp.Process(target=lossFunctionForCluster, args=(rawQueue, cookedQueue, hostIP, hostPort))
            worker.start()
        for worker in workerTank:
            worker.join()


    def showBestCoils(self):
        for coil in self.generation:
            print(coil.distribution)
            print(coil.loss)
            print('\n')
        coil = self.generation[0]
        # plotDistribution(coil.distributionInRealCoordinates, coil.minRadius, coil.Z0, points=100)


# Main

# # Ouer Coil
# r1 = 6.1e-2
# N1 = 1000
# l1 = 17.8e-2  # 20cm
# I1 = 1000
# R2 = 1e-7
# Ouer Coil
r1 = 30e-2
N1 = 1000
l1 = 20e-2  # 20cm
I1 = 1000
R2 = 1e-7


if __name__ == '__main__':
    mp.freeze_support()
    agent = GeneticAgent()

    modeString = sys.argv[1]
    if modeString == 'master' or modeString == 'm':
        agent.runAsMasterOnCluster()
    elif modeString == 'slave' or modeString == 's':
        agent.runAsSlaveOnCluster()
    agent.run()
    # agent.showBestCoils()
