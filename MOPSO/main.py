__author__ = 'tiago'

from swarm import Swarm, np
from pso import MOPSO
import function
from numpy import exp, where, sin, pi, sum, cos, log
import matplotlib.pyplot as plt

def PLOT(sol):

    x = np.arange(0.0, 1.0 , 0.001)
    y = 1 - x**2
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.title('ZDT6 FUNTION\nMOPSO Solutions')

    plt.plot(x,y,color = 'black')
    plt.plot(sol[:,0],sol[:,1],'ro')
    plt.show()

def main():

    SWARM       = Swarm(50,10,2,2,1,0)

    sol         = MOPSO(SWARM,function.ZDT6).Optimize()

    print SWARM.particles.bestPosition
    PLOT(sol)

if __name__ == "__main__": main()
