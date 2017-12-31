__author__ = 'tiago'

import numpy as np
from numpy import sum, copy, argsort, random, concatenate, logical_or, logical_and, all, any


MAX = 1.0

class Archive:

    def __init__(self,swarm):

        self.sizeOfObjective = swarm.sizeOfObjective
        self.sizeOfFitness   = swarm.sizeOfFitness
        self.sizeOfSwarm     = 2*swarm.sizeOfSwarm
        self.currFit         = swarm.particles.fitness
        self.currPos         = swarm.particles.position

        self.sizeOfComp      = int(0.05*self.sizeOfSwarm )

        self.pareto_opt = np.full(  (swarm.sizeOfSwarm,swarm.sizeOfFitness),10000000,dtype=np.float64)
        self.pareto_set = np.full(  (swarm.sizeOfSwarm,swarm.sizeOfDimension),0,dtype=np.float64)
        self.pareto_num = np.zeros( (swarm.sizeOfSwarm), dtype=np.int32)
        self.result     = np.zeros( (self.sizeOfSwarm,self.sizeOfSwarm),dtype=np.int32)
        self.set        = np.full(  (self.sizeOfSwarm,self.sizeOfSwarm+1),-1,dtype=np.int32)
        self.sfront     = np.zeros(self.sizeOfSwarm+1,dtype=np.int32)
        self.front      = np.zeros(self.sizeOfSwarm+1,dtype=np.int32)
        self.rank       = np.zeros(self.sizeOfSwarm,dtype=np.int32)
        self.dominated  = np.zeros(self.sizeOfSwarm,dtype=np.int32)
        self.density    = np.zeros(self.sizeOfSwarm,dtype=np.float64)
        self.fitness    = 0
        self.position   = 0

    def NonDominatedSort(self):

        k           = 1
        v           = 0
        density     = self.density
        set         = self.set
        sfront      = self.sfront
        front       = self.front
        rank        = self.rank
        dominated   = self.dominated
        fitness     = self.fitness = concatenate((self.currFit,self.pareto_opt))
        sizeOfSwarm = self.sizeOfSwarm
        result      = self.result
        position    = self.position = concatenate((self.currPos,self.pareto_set))
        pareto_opt  = self.pareto_opt
        pareto_set  = self.pareto_set
        pareto_num  = self.pareto_num

        dominated*=0 # vector is set to zero

        self.CrowdingDistance()

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

                if v < sizeOfSwarm/2:
                    pareto_opt[v] = copy(fitness[front[i]])
                    pareto_set[v] = copy(position[front[i]])
                    pareto_num[v] = front[i]

                x  = front[i]
                v += 1
                j  = 0

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


    def SelectionBest(self):

        rank        = self.rank
        density     = self.density
        position    = self.position
        pareto_num  = self.pareto_num
        n           = self.sizeOfComp
        sizeOfSwarm = self.sizeOfSwarm/2

        p    = pareto_num[random.randint(sizeOfSwarm, size = n)]
        best = p[0]

        for i in range(1,n):

            if density[best] == 0:
                best = p[i]
            elif rank[best] > rank[p[i]]:
                best = p[i]
            elif rank[best] == rank[p[i]] and density[best] < density[p[i]]:
                best = p[i]

        return position[best]


    def SelectionWost(self):

        rank        = self.rank
        density     = self.density
        position    = self.position
        pareto_num  = self.pareto_num
        n           = self.sizeOfComp
        sizeOfSwarm = self.sizeOfSwarm/2

        p    = pareto_num[random.randint(sizeOfSwarm, size = n)]
        wost = p[0]

        for i in range(1,n):

            if density[wost] < density[p[i]]:
                wost = p[i]

        return position[wost]

