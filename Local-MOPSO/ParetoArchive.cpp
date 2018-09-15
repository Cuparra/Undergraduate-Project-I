#include "Swarm.h"

#define TRUE        1
#define NEUTRAL     0
#define FALSE      -1

#define INF         1000

#define ABSOLUTE(num)  num = (num >= 0 ? num : -num)

#define CalculateTotalDensity(Total)    int tempSize = *CurrentSize;             \
                                        Total = 0;                              \
                                                                                \
                                        while(tempSize--)                       \
                                            Total += Leader[tempSize].Density;


#define Insert(x)       int LeaderIdx;                                      \
                        double density;                                         \
                        int include;                                        \
                                                                          \
                        if(*CurrentSize < SizeOfArchive){                      \
                            LeaderIdx   = ++*CurrentSize-1;                     \
                            include     = TRUE;                             \
                            density     = CalculateDensity(A, &particle[x], SizeOfObjective);       \
                        }else{                                              \
                            LeaderIdx   = FindCrowdedLeader(A);               \
                            density     = CalculateDensity(A, &particle[x], SizeOfObjective);       \
                            include     = density > Leader[LeaderIdx].Density;      \
                        }                                                           \
                                                                                        \
                        if(include){                                                  \
                                                                                        \
                            for(k=0;k<SizeOfFitness;k++)                                    \
                                Leader[LeaderIdx].Fitness[k] = particle[x].CurrentFitness[k];   \
                            for(k=0;k<SizeOfDimention;k++)                                 \
                                Leader[LeaderIdx].Position[k] = particle[x].CurrentPosition[k]; \
                            Leader[LeaderIdx].Density = density;                             \
                        }                                                                \
                        ++i;


#define Delete(x)       --*CurrentSize;                                     \
                        double *tempPos = Leader[x].Position;                \
                        double *tempFit = Leader[x].Fitness;              \
                                                                        \
                        for(k=x;k<SizeOfArchive-1;k++){                 \
                            Leader[k].Position = Leader[k+1].Position;      \
                            Leader[k].Fitness  = Leader[k+1].Fitness;       \
                        }                                               \
                                                                        \
                        Leader[SizeOfArchive-1].Position = tempPos;              \
                        Leader[SizeOfArchive-1].Fitness  = tempFit;


#define Dominate(x,y)   int XbestY       = 0;                                    \
                        int YbestX       = 0;                                    \
                        double *Yfitness = Leader[y].Fitness;                     \
                        double *Xfitness = particle[x].CurrentFitness;                     \
                                                                                \
                        for(k=0;k<SizeOfFitness;k++){                           \
                                                                                \
                            XbestY += (Xfitness[k] < Yfitness[k]);  \
                            YbestX += (Yfitness[k] < Xfitness[k]);  \
                        }                                                       \
                                                                                \
                        if(XbestY > 0 && YbestX == 0)                           \
                            result = TRUE;                                      \
                        else if (YbestX > 0 && XbestY == 0)                     \
                            result = FALSE;                                     \
                        else                                                    \
                            result = NEUTRAL;


#define SwapVec(x,y)        int temp = x; x = y; y = temp;


#define BubbleSort(vec,k)   int change = TRUE;                                                  \
                                                                                                \
                            while(change == TRUE){                                              \
                                                                                                \
                                change = FALSE;                                                 \
                                                                                                \
                                for (i = 0; i < CurrentSize - 1; i++){                          \
                                                                                                \
                                    int idx1 = vec[i];                                          \
                                    int idx2 = vec[i+1];                                        \
                                                                                                \
                                    if (Leader[idx1].Fitness[k] > Leader[idx2].Fitness[k]) {    \
                                        SwapVec(vec[i],vec[i+1]);                               \
                                        change = TRUE;                                          \
                                    }                                                           \
                                }                                                               \
                            }


int FindCrowdedLeader(Archive *A){

    int i;
    int Min                 = 0;
    ParetoParticle *Leader  = A->Leader;
    int CurrentSize         = A->CurrentSize;

    for(i=1;i<CurrentSize;i++){
        if(Leader[Min].Density > Leader[i].Density)
            Min = i;
    }

    return Min;
}

int RouletteWheel(Archive *A){

    int num                 = -1;
    double stopPoint        = 0.0;
    double TotalDensity     = A->TotalDensity;
    int CurrentSize         = A->CurrentSize;
    ParetoParticle *Leader  = A->Leader;
    double randomNumber     = ( (double) rand() / RAND_MAX);

    do{
        stopPoint += ( Leader[++num].Density / TotalDensity );
    }while(num < CurrentSize - 1 && stopPoint < randomNumber);

    return num;
}

static double CalculateDensity( Archive *A, Particle *p, int SizeOfObjective ){

    int i;
    int CurrentSize         = A->CurrentSize;
    ParetoParticle *Leader  = A->Leader;
    double density          = 0;

    for(i=0;i<SizeOfObjective;i++){
        int j       = 0;
        int *vec    = &A->VectorLeader[i][0]; //modificado

        while(j < CurrentSize && p->CurrentFitness[i] > Leader[vec[j]].Fitness[i])
            ++j;

        if(j == 0 || j == CurrentSize)
            density += INF;
        else{

            double result1 = Leader[vec[j]].Fitness[i] - Leader[vec[j-1]].Fitness[i];
            double result2 = Leader[vec[CurrentSize-1]].Fitness[i] + Leader[vec[0]].Fitness[i];
            density += ABSOLUTE(result1)/(ABSOLUTE(result2)+1);

        }
     }

    return density;
}

static void CrowdingDistance(Swarm *S, Archive *A){

    int i,j;
    ParetoParticle *Leader  = A->Leader;
    int CurrentSize         = A->CurrentSize;
    int SizeOfArchive       = S->SizeOfArchive;
    int SizeOfObjective     = S->SizeOfObjective;

    for(i=0 ;i<SizeOfArchive; i++)
        Leader[i].Density = 0;

    for(i=0;i<SizeOfObjective;i++){
        int *vec = A->VectorLeader[i];

        for(j=1;j<CurrentSize-1;j++){
            //modificado com +1 no final
            double result1 = Leader[vec[j+1]].Fitness[i] - Leader[vec[j-1]].Fitness[i];
            double result2 = Leader[vec[CurrentSize-1]].Fitness[i] + Leader[vec[0]].Fitness[i];
            Leader[vec[j]].Density += ABSOLUTE(result1)/(ABSOLUTE(result2)+1);
        }

        Leader[vec[0]].Density              += INF;
        Leader[vec[CurrentSize-1]].Density  += INF;
    }

}

static void CrowdingSort(Archive *A, int SizeOfObjective ){

    int i,k;
    ParetoParticle *Leader          = A->Leader;
    int CurrentSize                 = A->CurrentSize;
    int **VectorLeader              = A->VectorLeader;

    for(k=0;k<SizeOfObjective;k++){

        int *vec = &VectorLeader[k][0];
        BubbleSort(vec,k);

    }

}


void UpdateArchive(Swarm *S, Archive *A){

    int i = 0, j = 0, k;    //i index do S, j index do A

    ParetoParticle *Leader          = A->Leader;
    Particle *particle              = A->Particles;
    int *CurrentSize                = &A->CurrentSize;
    int SizeOfArchive               = S->SizeOfArchive;
    int SizeOfFitness               = S->SizeOfFitness;
    int SizeOfObjective             = S->SizeOfObjective;
    int SizeOfDimention             = S->SizeOfDimention;
    int SizeOfNeighbor              = S->SizeOfNeighbor;

    do{
        int result;
        Dominate(i,j);

        if(result == TRUE){

            Delete(j); //Inclui --CurrSize;

            if(j >= *CurrentSize){
                Insert(i); //Inclui ++i;
            }

        }else if(result == FALSE){
            j = 0;
            ++i;

        }else{
            if (j < *CurrentSize - 1)
                ++j;
            else{
                Insert(i); //Inclui ++i;
                j = 0;
            }
        }

    }while(i < SizeOfNeighbor);

    CrowdingSort(A,SizeOfObjective);
    CrowdingDistance(S,A);
    CalculateTotalDensity(A->TotalDensity);
}
