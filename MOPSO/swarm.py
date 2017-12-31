__author__ = 'tiago'

import numpy as np
from archive import Archive

INF = 65536.0

class Swarm:

    def __init__(self,sizeOfSwarm,sizeOfDimension,sizeOfFitness,sizeOfObjective,h_num,l_num):

        self.sizeOfDimension    = sizeOfDimension
        self.sizeOfSwarm        = sizeOfSwarm
        self.highestNumber      = h_num
        self.lowestNumber       = l_num
        self.sizeOfFitness      = sizeOfFitness    # constraint + objective
        self.sizeOfObjective    = sizeOfObjective

        self.particles          = Particles(self)
        self.archive            = Archive(self)

        self.DefaultParameter()


    def DefaultParameter(self):

        self.accelaration1      = 2.0
        self.accelaration2      = 2.0
        self.maxInteraction     = 300
        self.minWeight          = 0.4
        self.maxWeight          = 0.8
        self.weight             = self.minWeight + self.maxWeight


    def EspecifyParameter(self,acc1,acc2,maxInteraction,minWeight,maxWeight):

        self.accelaration1      = acc1
        self.accelaration2      = acc2
        self.maxInteraction     = maxInteraction
        self.minWeight          = minWeight
        self.maxWeight          = maxWeight - minWeight
        self.weight             = maxWeight


class Particles:

    def __init__(self,swarm):

        sizeofSwarm     = swarm.sizeOfSwarm
        sizeOfFitness   = swarm.sizeOfFitness
        l_num           = swarm.lowestNumber
        h_num           = swarm.highestNumber
        sizeOfDimension = swarm.sizeOfDimension

        self.currentFitness     = np.zeros( (sizeofSwarm , sizeOfFitness), np.float64)
        self.bestFitness        = np.full(  (sizeofSwarm, sizeOfFitness), INF, np.float64)
        self.leader             = np.zeros( (sizeofSwarm,sizeOfDimension),np.float64 )

        self.bestPosition       = np.zeros( (sizeofSwarm,sizeOfDimension), np.float64)
        self.velocity           = np.zeros( (sizeofSwarm,sizeOfDimension), np.float64)
        self.currentPosition    = np.random.rand(sizeofSwarm,sizeOfDimension)*(h_num-l_num) + l_num





