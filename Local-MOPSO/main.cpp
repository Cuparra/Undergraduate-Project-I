#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "ParticleSwarmOptimization.h"

/*Atenção: Colocar os objetivos primeiro, depois a restrição.*/
//modificar o INF para por aproximadamente o menor dos low ou high limit
//Cada particula pode usar gradient descend method

#define SIZE_P      100
#define SIZE_D      6
#define SIZE_F      8
#define SIZE_O      2
#define N_HI        10
#define N_LO        0
#define SESSION     10000

Swarm *S;

int session = 0;

void OsyczkaFunction(Particle *P){

    double *x = P->CurrentPosition;

    x[2] = 4*(x[2]/10) + 1;
    x[4] = 4*(x[4]/10) + 1;
    x[3] = 6*(x[3]/10);

    double fit1 = -25*(x[0]-2)*(x[0]-2) - (x[1]-2)*(x[1]-2) - (x[2]-1)*(x[2]-1) - (x[3]-4)*(x[3]-4) - (x[4]-1)*(x[4]-1);

    double fit2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + x[4]*x[4] + x[5]*x[5];

    double rest1 = x[0] + x[1] - 2;
    P->CurrentFitness[2] = rest1 = (rest1 >= 0) ? 0 : rest1*rest1;

    double rest2 = 6 - x[0] - x[1];
    P->CurrentFitness[3] = rest2 = (rest2 >= 0) ? 0 : rest2*rest2;

    double rest3 = 2 - x[1] + x[0];
    P->CurrentFitness[4] = rest3 = (rest3 >= 0) ? 0 : rest3*rest3;

    double rest4 = 2 - x[0] + 3*x[1];
    P->CurrentFitness[5] = rest4 =(rest4 >= 0) ? 0 : rest4*rest4;

    double rest5 = 4 - (x[2]-3)*(x[2]-3) - x[3];
    P->CurrentFitness[6] = rest5 = (rest5 >= 0) ? 0 : rest5*rest5;

    double rest6 = (x[4]-3)*(x[4]-3) + x[5] - 4;
    P->CurrentFitness[7] = rest6 = (rest6 >= 0) ? 0 : rest6*rest6;

    double restriction = rest1+rest2+rest3+rest4+rest5+rest6;

    P->CurrentFitness[0] = (restriction < 0.01) ? fit1 : 600;
    P->CurrentFitness[1] = (restriction < 0.01) ? fit2 : 600;

}

void EvaluateParticles(Particle * particles){

    int i;

    for(i=0;i<SIZE_P;i++)
        OsyczkaFunction(&particles[i]);

}

int TerminationCriteria(Swarm *S){

    int i,j;

    printf("%d\n",++session);

    if(session < SESSION)
        return 1;

    for(i=0;i<SIZE_P;i++){

        Archive *A              = S->ParetoFront;
        ParetoParticle *Leader  = A[i].Leader;
        int CurrentSize         = A[i].CurrentSize;

        for(j=0;j<CurrentSize;j++){
            double *fit = Leader[j].Fitness;
            printf("%lf %lf| %lf %lf %lf %lf %lf %lf\n",fit[0],fit[1],fit[2],fit[3],fit[4],fit[5],fit[6],fit[7]);
        }
        printf("\n");
        getchar();
     }

    getchar();

    return 0;
}

int main(){

    time_t t;
    srand((unsigned) time(&t));

    S = InicializeSwarm(SIZE_P,SIZE_D,SIZE_F,SIZE_O,2,2,N_HI,N_LO,SESSION);

    SwarmOptimization(S,EvaluateParticles,TerminationCriteria);

    return 0;
}
