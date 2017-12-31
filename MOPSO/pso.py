__author__ = 'tiago'

from swarm import np
import function
import matplotlib.pyplot as plt
NOTCHANGED = -1

class MOPSO:

    def __init__(self,swarm,function):

        self.time                = 0
        self.swarm               = swarm
        self.function            = function
        self.Comparator          = swarm.archive.Comparator
        self.maxInteraction      = swarm.maxInteraction
        self.TournamentSelection = swarm.archive.TournamentSelection
        self.NonDominatedSort    = swarm.archive.NonDominatedSort

    def Optimize(self):

        iteraction      = 0
        swarm           = self.swarm
        minWeight       = swarm.minWeight
        maxWeight       = swarm.maxWeight
        maxInteraction  = swarm.maxInteraction
        particles       = swarm.particles

        while self.TerminationCriteria():

            # Set the fitness of all particles
            self.EvaluateParticles(particles)

            # Update archive
            self.NonDominatedSort()

            # Set particle's best position
            self.SetParticleBestPosition(swarm)

            # Update the velocity and position of all particles
            self.UpdateParticles(swarm)

            # Update weight
            iteraction += 1
            swarm.weight = maxWeight*( 1.0*(maxInteraction - iteraction)/maxInteraction ) + minWeight

        return swarm.particles.bestFitness


    def UpdateParticles(self,swarm):

        w               = swarm.weight
        sizeOfSwarm     = swarm.sizeOfSwarm
        acc1            = swarm.accelaration1
        acc2            = swarm.accelaration2
        sizeOfDimension = swarm.sizeOfDimension

        particles = swarm.particles
        leader    = particles.leader
        velocity  = particles.velocity
        best      = particles.bestPosition
        curr      = particles.currentPosition

        for i in range(sizeOfSwarm):
            leader[i] = particles.bestPosition[self.TournamentSelection()]

        r1 = np.random.rand(sizeOfSwarm,sizeOfDimension)
        r2 = np.random.rand(sizeOfSwarm,sizeOfDimension)

        # This operation applies to the matrix
        velocity *= w
        velocity += acc1*r1*(best - curr) + acc2*r2*(leader - curr)
        curr     += velocity


    def SetParticleBestPosition(self,swarm):

        Comparator      = self.Comparator
        particles       = swarm.particles
        bestFit     = particles.bestFitness
        currFit  = particles.currentFitness
        currPos = particles.currentPosition
        bestPos    = particles.bestPosition

        sizeOfswarm     = swarm.sizeOfSwarm

        result = Comparator()

        for i in range(sizeOfswarm):
            if result[i] == True:
                bestFit[i] = 0 + currFit[i]
                bestPos[i] = 0 + currPos[i]


    def EvaluateParticles(self, particles):

        self.function( particles.currentFitness, particles.currentPosition )


    def TerminationCriteria(self):

        self.time += 1

        if self.time < self.maxInteraction:
            return True

        return False


    def Generate_Statistics(swarm,name,n_test,benchmark,efficacy,ref_point):

        fo = open(name + ".txt", "w")

        stat1 = np.zeros(n_test,np.float64)
        stat2 = np.zeros(n_test,np.float64)
        stat3 = np.zeros(n_test,np.float64)

        for i in range(n_test):

            sol         = MOPSO(swarm,benchmark).Optimize()

            stat1[i] = function.efficacy(sol,swarm.sizeOfSwarm)
            stat2[i] = function.HyperVolume(sol,swarm.sizeOfSwarm,ref_point)
            stat3[i] = function.Spacing(sol,swarm.sizeOfSwarm)

            print i, ": ", stat1[i], stat2[i], stat3[i]

        PrintStat(stat1)
        PrintStat(stat2)
        PrintStat(stat3)

        text = PrintStat(stat1) + PrintStat(stat2) + PrintStat(stat3)
        fo.write(text)
        fo.close()

    def PrintStat(stat):

        print "Mean = ", np.mean(stat)
        print "Var = ", np.var(stat)
        print "Min = ", np.amin(stat)
        print "Max = ", np.amax(stat),"\n"