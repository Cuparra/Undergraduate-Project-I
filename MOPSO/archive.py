__author__ = 'tiago'

import numpy as np
from numpy import sum, copy, argsort, random, concatenate, logical_or, logical_and, all, any


MAX = 1.0

class Archive:

    def __init__(self,swarm):

        self.sizeOfObjective = swarm.sizeOfObjective
        self.sizeOfFitness   = swarm.sizeOfFitness
        self.sizeOfSwarm     = 2*swarm.sizeOfSwarm
        self.currFit         = swarm.particles.currentFitness
        self.bestFit         = swarm.particles.bestFitness
        self.fitness         = concatenate(( self.bestFit, self.currFit))


        self.sizeOfComp      = int(0.1*swarm.sizeOfSwarm)

        self.result     = np.zeros( (self.sizeOfSwarm,self.sizeOfSwarm),dtype=np.int32)
        self.set        = np.full(  (self.sizeOfSwarm,self.sizeOfSwarm),-1,dtype=np.int32)
        self.sfront     = np.zeros(self.sizeOfSwarm+1,dtype=np.int32)
        self.front      = np.zeros(self.sizeOfSwarm+1,dtype=np.int32)
        self.rank       = np.zeros(self.sizeOfSwarm,dtype=np.int32)
        self.dominated  = np.zeros(self.sizeOfSwarm,dtype=np.int32)
        self.density    = np.zeros(self.sizeOfSwarm,dtype=np.float64)


    def NonDominatedSort(self):

        k       = 1

        set         = self.set
        sfront      = self.sfront
        front       = self.front
        rank        = self.rank
        dominated   = self.dominated
        fitness     = self.fitness = concatenate(( self.bestFit, self.currFit))
        sizeOfSwarm = self.sizeOfSwarm
        result      = self.result

        dominated*=0 # vector is set to zero

        # Set true or false whether particle i dominates or not j
        for i in range(sizeOfSwarm):
            result[i] = all( fitness[i] <= fitness, axis = 1)*any(fitness[i] < fitness, axis = 1)

        for i in range(sizeOfSwarm):

            # Find the particles that i dominates
            temp1 = np.where(result[i] == True)[0]
            # Put this particles in the set(i)
            set[i,:temp1.size] = temp1
            # End of set
            set[i][temp1.size] = -1
            # Calculate how many particles that dominate i
            dominated[i] = np.sum(result[:,i] == True)

        # Find all particles that are not dominated by anyone
        temp2 = np.where(dominated == 0)[0]
        # Put this particles in the first pareto front
        front[:temp2.size] = temp2
        # Rank the pareto front's particles to 1
        rank[temp2] = 1

        front[temp2.size] = -1
        count    = temp2.size

        while count:

            i = count = 0

            while front[i] != -1:

                x = front[i]
                j = 0

                while set[x][j] != -1:

                    y             = set[x][j]
                    dominated[y] -= 1
                    j            += 1

                    if dominated[y] == 0:
                        rank[y]         = k + 1
                        sfront[count]   = y
                        count           += 1
                i += 1

            k               += 1
            sfront[count]   = -1
            front           = copy(sfront)

        self.CrowdingDistance()


    def CrowdingDistance(self):

        sizeOfSwarm     = self.sizeOfSwarm
        sizeOfObjective = self.sizeOfObjective
        fit             = self.fitness
        density         = self.density

        density*=0

        for i in range(sizeOfObjective):

            x  = fit[:,i].argsort()

            b1 = ( fit[x[2:],i] - fit[x[0:sizeOfSwarm-2],i])**2
            b2 = ( fit[x[sizeOfSwarm-1],i] - fit[x[0],i])**2

            density[x[1:sizeOfSwarm-1]] += b1/(b2+1)
            density[x[0]]               += MAX
            density[x[sizeOfSwarm-1]]   += MAX


    def TournamentSelection(self):

        rank        = self.rank
        density     = self.density
        n           = self.sizeOfComp
        sizeOfSwarm = self.sizeOfSwarm / 2

        p    = random.randint(sizeOfSwarm, size = n)
        best = p[0]

        for i in range(1,n):

            if rank[best] > rank[p[i]]:
                best = p[i]
            elif rank[best] == rank[p[i]] and density[best] < density[p[i]]:
                best = p[i]

        return best


    def Comparator(self):

        n        = int(self.sizeOfSwarm/2)

        rank1    = self.rank[:n]
        rank2    = self.rank[n:]
        density1 = self.density[:n]
        density2 = self.density[n:]

        return logical_or(rank1 > rank2 , logical_and(rank1 == rank2, density1 < density2) )

