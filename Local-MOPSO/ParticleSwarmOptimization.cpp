#include "ParticleSwarmOptimization.h"

#define SIZE_ARCHIVE 10

#define  ALFA       0.975

#define MINIMUM     0.4
#define MAXIMUM     0.5

static double *GeneratePosition(int SizeOfDimention, double HighestNumber, double LowestNumber){

    double *Position = (double*)calloc(SizeOfDimention, sizeof(double));

    while(SizeOfDimention--)
        Position[SizeOfDimention]   = ( (double) rand() / RAND_MAX)*(HighestNumber-LowestNumber) + LowestNumber;

    return Position;

}

static void ConstructMesh(Swarm *S){

    int i,j;
    Archive *A          = S->ParetoFront;
    Particle *Particles = S->Particles;
    int SizeOfSwarm     = S->SizeOfSwarm;

    int SizeOfNeighbor  = sqrt(SizeOfSwarm);

    S->SizeOfNeighbor = 4;

    for(i=0; i<SizeOfSwarm;i++)
        Particles[i].Neighbor = (int*)malloc(4*sizeof(int));

    for(i=0; i<SizeOfNeighbor; i++){

        for(j=0; j<SizeOfNeighbor; j++){

            int pos = i*SizeOfNeighbor + j;

            Particles[pos].Neighbor[0] = (i-1)*SizeOfNeighbor + j;
            Particles[pos].Neighbor[1] = (i+1)*SizeOfNeighbor + j;
            Particles[pos].Neighbor[2] = pos - 1;
            Particles[pos].Neighbor[3] = pos + 1;

            if(i - 1 < 0)
               Particles[pos].Neighbor[0] = (SizeOfNeighbor - 1)*SizeOfNeighbor + j;

            if(i + 1 > SizeOfNeighbor - 1)
               Particles[pos].Neighbor[1] = j;

            if(pos - 1 < i*SizeOfNeighbor)
                Particles[pos].Neighbor[2] = (i+1)*SizeOfNeighbor - 1;

            if(pos + 1 > (i+1)*SizeOfNeighbor - 1)
                Particles[pos].Neighbor[3] = i*SizeOfNeighbor;

        }
    }

    for(i=0;i<SizeOfSwarm;i++){

        A[i].Particles = (Particle*)malloc(4*sizeof(Particle));

        for(j=0;j<4;j++){
            int idx           = Particles[i].Neighbor[j];
            A[i].Particles[j] = Particles[idx];
        }
    }

}

Swarm *InicializeSwarm(int SizeOfSwarm, int SizeOfDimention, int SizeOfFitness, int SizeOfObjective, double accelaration1, double accelaration2, double HighestNumber, double LowestNumber,  int MaxInteraction){

    int i,j,k;
    Swarm *S                = (Swarm*)malloc(sizeof(Swarm));
    S->SizeOfSwarm          = SizeOfSwarm;
    S->MaxInteraction       = MaxInteraction;
    S->SizeOfDimention      = SizeOfDimention;
    S->SizeOfFitness        = SizeOfFitness;
    S->accelaration1        = accelaration1;
    S->accelaration2        = accelaration2;
    S->weight               = MAXIMUM + MINIMUM;
    S->HighestNumber        = HighestNumber;
    S->LowestNumber         = LowestNumber;
    S->SizeOfObjective      = SizeOfObjective;
    S->SizeOfArchive        = 10;
    S->Particles            = (Particle*)calloc(SizeOfSwarm,sizeof(Particle));
    Particle *Particles     = S->Particles;

    for(i=0;i<SizeOfSwarm;i++){

        Particles[i].ParticleBestFitness     = (double*)malloc((SizeOfFitness)*sizeof(double));
        Particles[i].CurrentFitness          = (double*)malloc((SizeOfFitness)*sizeof(double));
        Particles[i].Velocity                = (double*)calloc(SizeOfDimention, sizeof(double));
        Particles[i].CurrentPosition         = GeneratePosition(SizeOfDimention,HighestNumber,LowestNumber);
        Particles[i].ParticleBestPosition    = (double*)calloc(SizeOfDimention, sizeof(double));

    }

    for(i=0;i<SizeOfSwarm;i++){

        for(j=0;j<SizeOfFitness;j++){

            Particles[i].ParticleBestFitness[j]     =  INT_MAX;
            Particles[i].CurrentFitness[j]          =  INT_MAX;
        }
    }

    S->ParetoFront = (Archive*)malloc(SizeOfSwarm*sizeof(Archive));

    for(i=0;i<SizeOfSwarm;i++){

        S->ParetoFront[i].CurrentSize           = SIZE_ARCHIVE;
        S->ParetoFront[i].Leader                = (ParetoParticle*)malloc(SIZE_ARCHIVE*sizeof(ParetoParticle));

        for(j=0; j<SIZE_ARCHIVE ;j++){

            S->ParetoFront[i].Leader[j].Fitness     = (double*)malloc(SizeOfFitness*sizeof(double));
            S->ParetoFront[i].Leader[j].Position    = (double*)malloc(SizeOfDimention*sizeof(double));

            for(k=0;k<SizeOfFitness;k++)
                S->ParetoFront[i].Leader[j].Fitness[k]   = INT_MAX;

            for(k=0;k<SizeOfDimention;k++)
                S->ParetoFront[i].Leader[j].Position[k]  = 0;
        }

        for(j=0;j<SIZE_ARCHIVE;j++)
            S->ParetoFront[i].Leader[j].Density = 0;

        S->ParetoFront[i].VectorLeader = (int**)malloc(SizeOfObjective*sizeof(int*));

        for(j=0;j<SizeOfObjective;j++)
            S->ParetoFront[i].VectorLeader[j] = (int*)malloc(SIZE_ARCHIVE*sizeof(int));

        for(j=0;j<SizeOfObjective;j++){
            for(k=0;k<SIZE_ARCHIVE;k++)
                S->ParetoFront[i].VectorLeader[j][k] = k;
        }

    }

    ConstructMesh(S);

    return S;

}

static inline void CopyPosition(double *Pos1, double *Pos2, int Size){

    while(Size--)
        Pos1[Size] = Pos2[Size];

}

/* Update the velocity and position*/
static void UpdateParticles(Swarm *S){

    int i,j;

    double w                = S->weight;
    Particle *particle      = S->Particles;
    Archive *A              = S->ParetoFront;
    double c1               = S->accelaration1;
    double c2               = S->accelaration2;
    double HighestNumber    = S->HighestNumber;
    double LowestNumber     = S->LowestNumber;
    int SizeOfDimention     = S->SizeOfDimention;
    int SizeOfSwarm         = S->SizeOfSwarm;

    for(i=0;i<SizeOfSwarm;i++){

        int LeaderIdx       = RouletteWheel(&A[i]);

        double *velocity    = particle[i].Velocity;
        double *gb          = A[i].Leader[LeaderIdx].Position;
        double *x           = particle[i].CurrentPosition;
        double *pb          = particle[i].ParticleBestPosition;

        for(j=0;j<SizeOfDimention;j++){

            double r1       = ( (double) rand()/RAND_MAX );
            double r2       = ( (double) rand()/RAND_MAX );

            velocity[j]     = w*velocity[j] + c1*r1*(pb[j] - x[j]) + c2*r2*(gb[j] - x[j]);

            x[j]            = x[j] + velocity[j];

            if(x[j] > HighestNumber)
                x[j] = HighestNumber;
            if(x[j] < LowestNumber)
                x[j] = LowestNumber;
        }
    }

}

static void SetParticleBestPosition(Swarm *S){

    int i,j;
    Particle *particle              = S->Particles;
    int SizeOfFitness               = S->SizeOfFitness;
    int SizeOfDimention             = S->SizeOfDimention;
    int SizeOfSwarm                 = S->SizeOfSwarm;

    for(i=0;i<SizeOfSwarm;i++){

        int cCurr           = 0;
        int cBest           = 0;
        double *CurrFitness = particle[i].CurrentFitness;
        double *BestFitness = particle[i].ParticleBestFitness;

        for(j=0;j<SizeOfFitness;j++){

            cCurr += (CurrFitness[j] < BestFitness[j]);
            cBest += (BestFitness[j] < CurrFitness[j]);

        }

        if(cBest > 0 && cCurr == 0){

            for(j=0;j<SizeOfFitness;j++)
                particle[i].ParticleBestFitness[j] = particle[i].CurrentFitness[j];

            CopyPosition(particle[i].ParticleBestPosition, particle[i].CurrentPosition,SizeOfDimention);
        }

    }
}

static void SetNeighborhoodBestPosition(Swarm *S){

    int i;
    Archive *A      = S->ParetoFront;
    int SizeOfSwarm = S->SizeOfSwarm;

    for(i=0;i<SizeOfSwarm;i++)
        UpdateArchive(S,&A[i]);

}

void SwarmOptimization(Swarm *S, void (*EvaluateParticles)(Particle *), int (*TerminationCriteria)(Swarm *)){
	
    int Cycle		    = 0;
    int MaxInteraction  = S->MaxInteraction;
    Particle *Particles = S->Particles;

    do{

        /*Set the fitness of all particles*/
        EvaluateParticles(Particles);

        /*Set particle's best position*/
        SetParticleBestPosition(S);

        /*Set the neighborhood(local) best position */
        SetNeighborhoodBestPosition(S);

        /* Update the velocity and position of all particles*/
        UpdateParticles(S);

        /*Update weight and sigma*/
        ++Cycle;
        S->weight  = MAXIMUM*( 1.0*(MaxInteraction - Cycle)/MaxInteraction ) + MINIMUM;

    }while(TerminationCriteria(S));

}
