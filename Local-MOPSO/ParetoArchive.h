#ifndef PARETOARCHIVE_H
#define PARETOARCHIVE_H

#include "Swarm.h"
/*
typedef struct PARETOPARTICLE{
    double *Position;
    double *Fitness;
    double Density;
}ParetoParticle;

typedef struct ARCHIVE{
    int CurrentSize;
    int **VectorLeader;
    int SizeOfArchive;
    double TotalDensity;
    ParetoParticle *Leader;
}Archive;*/

//void UpdateArchive(Swarm *, int);
int TournamentSelection(Archive *);
//int RouletteWheel(Archive *);

#endif
