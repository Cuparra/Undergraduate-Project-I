#ifndef CONTINUOUSPARTICLESWARMOPT_H
#define CONTINUOUSPARTICLESWARMOPT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "Swarm.h"

#define NOTCHANGED      -1

/* -10<=V<=10 */
#define sigmoid(V,M)            M/(1.0 + exp(-V))

/*Size Of Population, Size Of Dimention, Size Of Fitness, Size Of Objectives, accelaration1, accelaration2, HighestNumber, LowestNumber, max interaction,*/
Swarm *InicializeSwarm(int , int , int, int ,  double, double, double, double, int);

/*Population, EvaluateParticle, TerminationCriteria*/
void SwarmOptimization(Swarm *, void (*)(Particle *), int (*)(Swarm *));

//Particle *ContinuousBestGlobalParticle(Population *);
#endif
