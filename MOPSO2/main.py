__author__ = 'tiago'

from swarm import Swarm, np
from pso import MOPSO
import function
from numpy import exp, where, sin, pi, sum, cos, log
import matplotlib.pyplot as plt
from multiprocessing import Process

# FALTA limitar as variaveis em l_num e h_num

def PLOT(sol):

    x = np.arange(0.0, 1.0 , 0.001)
    y = 1 - x**0.5
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.title('ZDT6 FUNTION\nMOPSO Solutions')

    plt.plot(x,y,color = 'black')
    plt.plot(sol[:,0],sol[:,1],'ro')
    plt.show()

def PrintStat(stat):

        x = np.mean(stat)
        y = np.var(stat)
        z = np.amax(stat) - np.amin(stat)
        print "Mean = ", x
        print "Var = ", y
        print "Range = ", z, "\n"

        return "Mean = " + x + "\nVar = " + y + "Range = " + z + "\n"

def Generate_Statistics(swarm,name,n_test,benchmark,efficacy,ref_point):

        fo = open(name+".txt","w")

        stat1 = np.zeros(n_test,np.float64)
        stat2 = np.zeros(n_test,np.float64)
        stat3 = np.zeros(n_test,np.float64)

        sizeOfSwarm = swarm.sizeOfSwarm
        dimention   = swarm.sizeOfDimension
        fit        = swarm.sizeOfFitness
        obj        = swarm.sizeOfObjective
        h_num       = swarm.highestNumber
        l_num       = swarm.lowestNumber

        for i in range(n_test):

            SWARM       = Swarm(sizeOfSwarm,dimention,fit,obj,h_num,l_num)
            sol         = MOPSO(SWARM,benchmark).Optimize()

            stat1[i] = efficacy(sol,sizeOfSwarm)
            stat2[i] = function.HyperVolume(sol,sizeOfSwarm,ref_point)
            stat3[i] = function.Spacing(sol,sizeOfSwarm)

            print name, i, ": ", stat1[i], stat2[i], stat3[i]
            print sol
        print name

        text = PrintStat(stat1) + PrintStat(stat2) + PrintStat(stat3)
        fo.write(text)
        fo.close()

def main():

    SWARM(50,30,function.ZDT1,2,2,1.0,0.0)
    sol = MOPSO(SWARM,function.ZDT1)

if __name__ == "__main__": main()
