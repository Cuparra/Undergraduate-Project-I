
import numpy as np
from numpy import exp, where, sin, pi, sum, cos, fabs
INF = 65536.0

def ZDT1(fit, x):

    x[x>1]  = 1
    x[x<0] = 0

    g = 1 + (9.0/29)*sum(x[:,1:],axis = 1)
    h = 1 - (x[:,0]/g[0:])**0.5

    fit[:,0] = x[:,0]
    fit[:,1] = g[0:]*h[0:]


def ZDT2(fit, x):

    x[x>1]  = 1
    x[x<0] = 0

    g = 1 + (9.0/29)*sum(x[:,1:],axis = 1)
    h = 1 - (x[:,0]/g[0:])**2

    fit[:,0] = x[:,0]
    fit[:,1] = g[0:]*h[0:]


def ZDT3(fit,x):

    x[x>1] = 1
    x[x<0] = 0

    g = 1 + (9.0/29)*sum(x[:,1:],axis = 1)
    h = 1 - (x[:,0]/g[0:])**0.5 - (x[:,0]/g[0:])*sin(10*pi*x[:,0])

    fit[:,0] = x[:,0]
    fit[:,1] = g[0:]*h[0:]

def ZDT4(fit, x):

    size = np.size(x, axis= 1)

    x[x>1] = 1
    x[x<0] = 0

    g = 1 + 10*(size - 1) + sum( (x[:,1:]*(10.0 - 5))**2  - 10*cos(4*pi*x[:,1:]*(10.0 - 5)) ,axis = 1)
    h = 1 - (x[:,0]/g[0:])**0.5

    fit[:,0] = x[:,0]
    fit[:,1] = g[0:]*h[0:]


def ZDT6(fit,x):

    x[x>1]  = 1
    x[x<0] = 0

    g = 1 + 9*(sum(x[:,1:],axis = 1)/9)**0.25

    fit[:,0] = 1 - exp(-4*x[:,0])*sin(6*pi*x[:,0])**6
    fit[:,1] = 1 - (fit[:,0]/g[0:])**2


def FON(fit,x):

    x[x>2]  = 2
    x[x<-2] = -2

    con = 1.0/(8**0.5)

    fit[:,0] = 1 - exp( -sum((x[:,0:] - con)**2, axis = 1) )
    fit[:,1] = 1 - exp( -sum((x[:,0:] + con)**2, axis = 1) )

def problem1(fit,x):

    x[x>1] = 1
    x[x<-1] = -1

    for i in range(50):

        sum1 = 0
        sum2 = 0

        for j in range(2,30,2):
            sum1 +=(x[i][j] + sin(6*pi*x[i][j] + ((j+1)*pi)/30.0 ))**2
        for j in range(1,30,2):
            sum2 += (x[i][j] + sin(6*pi*x[i][j] + (( j+1 )*pi)/30.0 ))**2

        x[i][0]   = (x[i][0]+1.0)/2.0
        fit[i][0] = x[i][0] + (2.0/15)*sum1
        fit[i][1] = 1 - x[i][0]**0.5 + (2.0/15)*sum2


def efficacyZDT1(sol,sizeOfSwarm):

    result1 = (1 - sol[:,0]**0.5 - sol[:,1])**2
    result2 = sum(where(result1 <= 0.00001, 1, 0))

    return 1.0*result2/sizeOfSwarm

def efficacyZDT2(sol,sizeOfSwarm):

    result1 = (1 - sol[:,0]**2 - sol[:,1])**2
    result2 = sum(where(result1 <= 0.00001, 1, 0))

    return 1.0*result2/sizeOfSwarm

def efficacyZDT3(sol, sizeOfSwarm):

        result1      = (1 - sol[:,0]**0.5 - sol[:,0]*sin(10*pi*sol[:,0]) - sol[:,1])**2

        logic1       =  np.logical_and(sol[:,0] >= 0, sol[:,0] <= 0.08300154)
        logic2       =  np.logical_and(sol[:,0] > 0.18222873, sol[:,0] <= 0.25776236)
        logic3       =  np.logical_and(sol[:,0] > 0.40931367, sol[:,0] <= 0.45388211)
        logic4       =  np.logical_and(sol[:,0] > 0.61839679, sol[:,0] <= 0.65251171)
        logic5       =  np.logical_and(sol[:,0] > 0.82333180, sol[:,0] <= 0.85183286)

        total          =  logic1+logic2+logic3+logic4+logic5

        result2      = np.sum(np.logical_and( result1 <= 0.1, total))

        statistic   = 1.0*result2/sizeOfSwarm

        return statistic

def efficacyZDT4(sol,sizeOfSwarm):

    result1 = (1 - sol[:,0]**0.5 - sol[:,1])**2
    result2 = sum(where(result1 <= 0.00001, 1, 0))

    return 1.0*result2/sizeOfSwarm

def efficacyZDT6(sol,sizeOfSwarm):

    result1 = (1 - sol[:,0]**2 - sol[:,1])**2
    result2 = sum(where(result1 <= 0.00001, 1, 0))

    return 1.0*result2/sizeOfSwarm

def efficacyFON(x,sizeOfSwarm):

    con = 1.0/(8**0.5)

    result = sum(np.all( np.logical_and(x>= -con, x <= con) , axis = 1))

    return 1.0*result/sizeOfSwarm

def efficacyPROBLEM1(sol,sizeOfSwarm):

    result1 = (1 - sol[:,0]**0.5 - sol[:,1])**2
    result2 = sum(where(result1 <= 0.00001, 1, 0))

    return 1.0*result2/sizeOfSwarm

def Spacing(fit, n):


    num = 10**20
    x   = np.zeros( (n,n), np.float64)

    for i in range(n):
        x[i]    = np.sum((fit - fit[i])**2, axis = 1)
        x[i][i] = INF

    mini  = np.min(x,axis = 1)

    mean = 1.0*np.sum(mini)/n

    s = num*( (np.sum(mini - mean)**2 )  /n )**0.5

    return s

def HyperVolume(fit,n,point):

    x  = fit[:,0].argsort()

    p1 = point[0]
    p2 = point[1]

    s = sum(( fabs(fit[x[1:],0] - fit[x[0:n-1],0]) )*( fabs(p2 - fit[x[0:n-1],1]) ))
    s += ( fabs(p1 - fit[x[n-1],0]) )*( fabs(p2 - fit[x[n-1],1]) )

    return s