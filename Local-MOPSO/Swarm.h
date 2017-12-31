#ifndef POPULATION_H
#define POPULATION_H

#include <stdlib.h>

typedef struct PARTICLE{
    int *Neighbor;
    double *Velocity;

    double *CurrentFitness;
    double *CurrentPosition;

    double *ParticleBestFitness;
    double *ParticleBestPosition;

}Particle;

typedef struct PARETOPARTICLE{
    double *Position;
    double *Fitness;
    double Density;
}ParetoParticle;

typedef struct ARCHIVE{
    int CurrentSize;
    int **VectorLeader;
    double TotalDensity;
    Particle *Particles;
    ParetoParticle *Leader;
}Archive;

typedef struct SWARM{
    Archive *ParetoFront;
    Particle *Particles;
    Particle *BestParticle;
    int SizeOfNeighbor;
    int SizeOfFitness;      /*It includes objectives and constraints*/
    int SizeOfObjective;    /*Includes objectives only*/
    int SizeOfArchive;
    int SizeOfDimention;
    int SizeOfSwarm;
    double accelaration1;
    double accelaration2;
    int MaxInteraction;
    double weight;
    double Range;
    double HighestNumber;
    double LowestNumber;
}Swarm;

void UpdateArchive(Swarm *, Archive *);
int RouletteWheel(Archive *);

#endif
