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
        self.archive             = swarm.archive
        self.maxInteraction      = swarm.maxInteraction
        self.SelectionBest       = swarm.archive.SelectionBest
        self.SelectionWost       = swarm.archive.SelectionWost
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

            # Update the velocity and position of all particles
            self.UpdateParticles(swarm)

            # Update weight
            iteraction += 1
            swarm.weight = 0.7*( 1.0*(maxInteraction - iteraction)/maxInteraction ) + minWeight

        return swarm.archive.pareto_opt


    def UpdateParticles(self,swarm):

        w               = swarm.weight
        sizeOfSwarm     = swarm.sizeOfSwarm
        acc1            = swarm.accelaration1
        acc2            = swarm.accelaration2
        sizeOfDimension = swarm.sizeOfDimension

        particles = swarm.particles
        best      = particles.best
        wost      = particles.wost
        velocity  = particles.velocity
        pos       = particles.position

        for i in range(sizeOfSwarm):
            best[i] = self.SelectionBest()
            wost[i] = self.SelectionWost()

        r1   = np.random.rand(sizeOfSwarm,sizeOfDimension)
        r2   = np.random.rand(sizeOfSwarm,sizeOfDimension)

        # This operation applies to the matrix
        velocity *= w
        velocity += acc1*r1*(best - pos)
        pos     += velocity


    def EvaluateParticles(self, particles):

        self.function( particles.fitness, particles.position )


    def TerminationCriteria(self):

        self.time += 1

        if self.time < self.maxInteraction:
            return True

        return False


    def PrintStat(stat):

        print "Mean = ", np.mean(stat)
        print "Var = ", np.var(stat)
        print "Min = ", np.amin(stat)
        print "Max = ", np.amax(stat),"\n"


    def Generate_Statistics(swarm,n_test,benchmark,efficacy,ref_point):

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

