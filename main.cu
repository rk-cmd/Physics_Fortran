#include <iostream>
#include <cuda.h>

#include <unistd.h>
#include <math.h>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define THREADS_PER_BLOCK 64

#define h_NX 1001
#define h_NY2 401
#define h_NY1 401
#define h_NTIMES 5000

__device__ double UINF, PINF, TINF, PI;
__device__ double XL, THETA, RS, ROEEP, GAMMAN, CFL;
__device__ int NX, NY1, NTIMES;
__device__ float ERROR;

// DOUBLE PRECISION :: TTOT, PTOT, RHOTOT, TOUT, POUT, RHOOUT, COUT, UOUT, RHOINF,DT,DTB,VOL
// DOUBLE PRECISION :: YL1, YL2, RTHETA, DX, DY, MINF, CINF, LH, M2, M2N, FX, FXX           
// DOUBLE PRECISION :: UUJ, CC, XX, PPF, UUJP, UUJN, DDT, RATIO,DT1
// INTEGER :: I, J, K, L, N, KM, NY2, EXNY, ITER, REMAIN, II,JJ,ITER2, ICOUNT

__device__ double TTOT, PTOT, RHOTOT, TOUT, POUT, RHOOUT, COUT, UOUT, RHOINF, DT, DTB, VOL;
__device__ double YL1, YL2, RTHETA, DX, DY, MINF, CINF, LH, M2, M2N, FX, FXX;
__device__ double UUJ, CC, XX, PPF, UUJP, UUJN, DDT, RATIO, DT1;
__device__ int I, J, K, L, N, KM, NY2, EXNY, ITER, REMAIN, II,JJ,ITER2, ICOUNT;

        //    DOUBLE PRECISION, ALLOCATABLE :: P(:,:), U(:,:), V(:,:), RHO(:,:), RHOU(:,:), RHOV(:,:), E(:,:),SOS(:,:)
        //    DOUBLE PRECISION, ALLOCATABLE :: DFI(:,:,:), DFJ(:,:,:), DFF(:,:,:), DFE(:,:,:), DU(:,:,:), MACH(:,:) 
        //    DOUBLE PRECISION, ALLOCATABLE :: T(:,:), YN(:,:), XN(:,:)

// __device__ double *P, *U, *V, *RHO, *RHOU, *RHOV, *E, *SOS;
// __device__ double *DFI, *DFJ, *DFF, *DFE, *DU, *MACH;
// __device__ double *T, *YN, *XN;

__device__ double U[h_NX][h_NY2], V[h_NX][h_NY2], P[h_NX][h_NY2], E[h_NX][h_NY2];
__device__ double RHO[h_NX][h_NY2], RHOU[h_NX][h_NY2], RHOV[h_NX][h_NY2];
__device__ double DFI[h_NX][h_NY2][4], DFJ[h_NX][h_NY2][4], DFF[h_NX][h_NY2][4];
__device__ double DFE[h_NX][h_NY2][4], DU[h_NX][h_NY2][4];
__device__ double SOS[h_NX][h_NY2], MACH[h_NX][h_NY2];
__device__ double T[h_NX][h_NY2], XN[h_NX][h_NY2], YN[h_NX][h_NY2];

double h_U[h_NX][h_NY2], h_V[h_NX][h_NY2], h_P[h_NX][h_NY2], h_RHO[h_NX][h_NY2], h_MACH[h_NX][h_NY2];
//  U[I][J], V[I][J], P[I][J], RHO[I][J], MACH[I][J]


// CUDA kernel to add two numbers
__global__ void add(int a, int b, int *result) {
    *result = a + b;
}

// some parts of subroutine GEOMETRY infused here
__global__ void global_variable_init() {
    
    UINF = 300.0;
    PINF = 101325.0;
    TINF = 288.0;
    PI = 4.0*atan(1.0);

    XL = 0.5;
    THETA = 14.0632;
    RS = 287.0;
    ROEEP = 1.0e-3;
    GAMMAN = 1.4;
    CFL = 0.1;    

    NX = 1001;
    NY1 = 401;
    NTIMES = 5000;

    ERROR = 0.00001f;

    // __________INLET VARIABLES_______________________________
    CINF = sqrt(GAMMAN*RS*TINF);
    MINF = UINF/CINF;
    RHOINF = PINF/(RS*TINF);
    // __________________________STAGNATION CONDITION______________________________    
    TTOT = TINF * (1.0 + 0.5 * (GAMMAN - 1.0 ) * pow(MINF, 2));
    PTOT = PINF / pow((TINF / TTOT), (GAMMAN / (GAMMAN - 1.0)));
    RHOTOT = RHOINF * pow(1.0 + 0.5 * (GAMMAN - 1.0) * pow(MINF, 2), 1.0 / (GAMMAN - 1.0));

}

// __global__ void memory_allocation(double *block_ptr) {

//        //    ALLOCATE(U(NX,NY2),V(NX,NY2),P(NX,NY2),E(NX,NY2))
//         //    ALLOCATE(RHO(NX,NY2),RHOU(NX,NY2),RHOV(NX,NY2))
//         //    ALLOCATE(DFI(NX,NY2,4),DFJ(NX,NY2,4),DFF(NX,NY2,4))
//         //    ALLOCATE(DFE(NX,NY2,4), DU(NX,NY2,4))
//         //    ALLOCATE(SOS(NX,NY2),MACH(NX,NY2))
//         //    ALLOCATE(T(NX,NY2))

// // __device__ double U[NX][NY2], V[NX][NY2], P[NX][NY2], E[NX][NY2];
// // __device__ double RHO[NX][NY2], RHOU[NX][NY2], RHOV[NX][NY2];
// // __device__ double DFI[NX][NY2][4], DFJ[NX][NY2][4], DFF[NX][NY2][4];
// // __device__ double DFE[NX][NY2][4], DU[NX][NY2][4];
// // __device__ double SOS[NX][NY2], MACH[NX][NY2];
// // __device__ double T[NX][NY2];


//     U = block_ptr;
//     V = block_ptr + (1 * NX *NY2);
//     P = block_ptr + (2 * NX *NY2);
//     E = block_ptr + (3 * NX *NY2);

//     RHO = block_ptr + (4 * NX *NY2);
//     RHOU = block_ptr + (5 * NX *NY2);
//     RHOV = block_ptr + (6 * NX *NY2);

//     SOS = block_ptr + (7 * NX *NY2);
//     MACH = block_ptr + (8 * NX *NY2);
//     T = block_ptr + (9 * NX *NY2);

//     DFI = block_ptr + (10 * NX *NY2);
//     DFJ = block_ptr + (14 * NX *NY2);
//     DFF = block_ptr + (18 * NX *NY2);

//     DFE = block_ptr + (22 * NX *NY2);
//     DU = block_ptr + (26 * NX *NY2);

//     XN = block_ptr + (30 * NX *NY2);
//     YN = block_ptr + (30 * NX *NY2) + (NX * NY1);

// }

__global__ void geometry_init() {

    DX = XL / (NX - 1);
    DY = DX;
    YL1 = DY * (NY1 - 1);
    NY2 = NY1;

}

__global__ void geometry_prefix_sum_1() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < NY1) {

        for(unsigned long i = 1 ; i < NX ; i++)
            XN[i][id] = XN[i - 1][id] + DX;

    }

}

__global__ void geometry_prefix_sum_2() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < NX) {

        for(unsigned long i = 1 ; i < NY1 ; i++)
            YN[id][i] = YN[id][i - 1] + DY;

    }

}

__global__ void geometry_end() {

    VOL = DX * DY;
    DT = CFL * DX / (UINF + CINF);
    // printf("At geometry_end, DT is %.16f\n", DT);

}

void geometry() {

    geometry_init<<<1, 1>>>();

    int thread_blocks = ceil(double(h_NY1) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();
    geometry_prefix_sum_1<<<thread_blocks, THREADS_PER_BLOCK>>>();

    thread_blocks = ceil(double(h_NX) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();
    geometry_prefix_sum_2<<<thread_blocks, THREADS_PER_BLOCK>>>();

    geometry_end<<<1, 1>>>();
    cudaDeviceSynchronize();

}

__global__ void initial_condition() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (NX * NY2)) {

        unsigned long i = id / NY2;
        unsigned long j = id % NY2;

        U[i][j] = 0.0;
        V[i][j] = 0.0;
        P[i][j] = PINF;
        RHO[i][j] = RHOINF;
        RHOU[i][j] = RHO[i][j] * U[i][j];
        RHOV[i][j] = RHO[i][j] * V[i][j];
        E[i][j] = P[i][j] / (GAMMAN - 1.0) + 0.5 * RHO[i][j] * (pow(U[i][j], 2) + pow(V[i][j], 2));    

    }

}

__global__ void inlet_boundary_condition() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < NY2) {

        U[0][id] = UINF;
        V[0][id] = 0.0;
        P[0][id] = PINF;
        RHO[0][id] = RHOINF;
        RHOU[0][id] = RHO[0][id] * U[0][id];
        RHOV[0][id] = RHO[0][id] * V[0][id];
        E[0][id] = P[0][id] / (GAMMAN - 1.0) + 0.5 * RHO[0][id] * (pow(U[0][id], 2) + pow(V[0][id], 2));     

    }

}

void initial() {

    int thread_blocks = ceil(double(h_NX * h_NY2) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();
    initial_condition<<<thread_blocks, THREADS_PER_BLOCK>>>();
    
    thread_blocks = ceil(double(h_NY2) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();
    inlet_boundary_condition<<<thread_blocks, THREADS_PER_BLOCK>>>();    
    cudaDeviceSynchronize();

}

__global__ void lower_wall() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

// id should run from 1 to NX-2

    if(((id < (NX - 2)))) {

        id++;

        unsigned long j = 0;
        // U[id][j] = U[id][j + 1];


        U[id][j] = U[id][j + 1];
        V[id][j] = V[id][j + 1];
        P[id][j] = P[id][j + 1];
        RHO[id][j] = RHO[id][j + 1];
        RHOU[id][j] = RHO[id][j] * U[id][j];
        RHOV[id][j] = RHO[id][j] * V[id][j];
        E[id][j] = E[id][j + 1];

    }

    // if(!id)
    //     printf("Device at lower_wall for [1][0], RHO[1][1]=%.16f\n", RHO[1][1]);

}

__global__ void upper_wall() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(((id < (NX - 2)))) {

        id++;

        unsigned long j = NY2 - 1;

        U[id][j] = U[id][j - 1];
        V[id][j] = V[id][j - 1];
        P[id][j] = P[id][j - 1];
        RHO[id][j] = RHO[id][j - 1];
        RHOU[id][j] = RHO[id][j] * U[id][j];
        RHOV[id][j] = RHO[id][j] * V[id][j];
        E[id][j] = E[id][j - 1];

    }

}

__global__ void outlet_boundary_condition() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (NY2)) {

        unsigned long i = NX - 1;

        U[i][id] = U[i - 1][id];
        V[i][id] = V[i - 1][id];
        P[i][id] = PINF;
        RHO[i][id] = RHO[i - 1][id];
        RHOU[i][id] = RHOU[i - 1][id];
        RHOV[i][id] = RHOV[i - 1][id];
        E[i][id] = P[i][id] / (GAMMAN - 1) + 0.5 * RHO[i][id] * (pow(U[i][id], 2) + pow(U[i][id], 2));

    }

}

__global__ void outlet_boundary_condition_end() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (NX * NY2)) {

        unsigned long i = id / NY2;
        unsigned long j = id % NY2;

        T[i][j] = P[i][j] / (RS * RHO[i][j]);

    }

}

void bc() {

    int thread_blocks = ceil(double(h_NX - 2) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();
    lower_wall<<<thread_blocks, THREADS_PER_BLOCK>>>();    
    upper_wall<<<thread_blocks, THREADS_PER_BLOCK>>>();    
    cudaDeviceSynchronize();
    thread_blocks = ceil(double(h_NY2) / THREADS_PER_BLOCK);
    outlet_boundary_condition<<<thread_blocks, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
    thread_blocks = ceil(double(h_NX * h_NY2) / THREADS_PER_BLOCK);
    outlet_boundary_condition_end<<<thread_blocks, THREADS_PER_BLOCK>>>();

}

__global__ void FXI() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 1204)
    //     printf("Before FXI %lu %lu %.16f %.16f %.16f %.16f %.16f\n", id / NY2, id % NY2, DFJ[3][1][2], RHOV[3][1], 0.5 * (U[3][1] - (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1]))), U[3][1], (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1])), sqrt(GAMMAN * P[3][1] / RHO[3][1]));

    if(id < (NX * NY2)) {

        unsigned long i = id / NY2;
        unsigned long j = id % NY2;

        double UUJ = U[i][j];
        double CC = sqrt(GAMMAN * P[i][j] / RHO[i][j]);
        double XX = (sqrt(pow((ROEEP * CC), 2) + pow(UUJ, 2)) + CC);
    
        double PPF  = 0.5 * P[i][j];
        double UUJP = 0.5 * (UUJ + XX);
        double UUJN = 0.5 * (UUJ - XX);

        DFI[i][j][0] = (RHO[i][j]) * UUJP;
        DFI[i][j][1] = (RHOU[i][j]) * UUJP + PPF;
        DFI[i][j][2] = (RHOV[i][j]) * UUJP;
        DFI[i][j][3] = (E[i][j]) * UUJP + PPF * UUJ;

        DFJ[i][j][0] = (RHO[i][j]) * UUJN;
        DFJ[i][j][1] = (RHOU[i][j]) * UUJN + PPF;
        DFJ[i][j][2] = (RHOV[i][j]) * UUJN;
        DFJ[i][j][3] = (E[i][j]) * UUJN + PPF * UUJ;

    }    

    // if(id == 1204) {
    //     printf("At FXI CC, GAMMAN=%.16f, P=%.16f, RHO=%.16f\n", GAMMAN, P[3][1], RHO[3][1]);
    //     printf("After FXI %lu %lu UUJN=%.16f %.16f %.16f %.16f %.16f CC=%.16f\n", id / NY2, id % NY2, DFJ[3][1][2], RHOV[3][1], 0.5 * (U[3][1] - (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1]))), U[3][1], (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1])), sqrt(GAMMAN * P[3][1] / RHO[3][1]));
    // }
    // if(id == 1605) {
    //     printf("At FXI CC, GAMMAN=%.16f, P=%.16f, RHO=%.16f\n", GAMMAN, P[4][1], RHO[4][1]);
    //     printf("After FXI %lu %lu %.16f %.16f UUJN=%.16f %.16f %.16f CC=%.16f\n", id / NY2, id % NY2, DFJ[4][1][2], RHOV[4][1], 0.5 * (U[3][1] - (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1]))), U[3][1], (sqrt(pow((ROEEP * sqrt(GAMMAN * P[3][1] / RHO[3][1])), 2) + pow(U[3][1], 2)) + sqrt(GAMMAN * P[3][1] / RHO[3][1])), sqrt(GAMMAN * P[3][1] / RHO[3][1]));

    // }
}

// __global__ void DFXI() {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if(id < ((NX - 2) * (NY2 - 2))) {

//         unsigned long i = id / (NY2 - 2);
//         unsigned long j = id % (NY2 - 2);

//         for(unsigned long n = 0 ; n < 4 ; n++)
//             DFF[i + 1][j + 1][n] = -(DFI[i + 1][j + 1][n] - DFI[i][j + 1][n]) / DX - (DFJ[i + 2][j + 1][n] - DFJ[i + 1][j + 1][n]) / DX;

//     }    

// }

__global__ void DFXI() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (4 * (NY2 - 2))) {

        unsigned long n = id / (NY2 - 2);
        // unsigned long i = id / (NY2 - 2);
        unsigned long j = id % (NY2 - 2);

        for(unsigned long i = 0 ; i < NX - 2 ; i++)
            DFF[i + 1][j + 1][n] = -(DFI[i + 1][j + 1][n] - DFI[i][j + 1][n]) / DX - (DFJ[i + 2][j + 1][n] - DFJ[i + 1][j + 1][n]) / DX;

    }    

    // if(id == 798)
    //     printf("At DFXI, n=%lu j=%lu, %.16f %.16f %.16f %.16f %.16f DX=%.16f\n", id / (NY2 -2), id % (NY2 -2), DFF[4][1][2], DFI[4][1][2], DFI[3][1][2], DFJ[5][1][2], DFJ[4][1][2], DX);

}

__global__ void FYI() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (NX * NY2)) {

        unsigned long i = id / NY2;
        unsigned long j = id % NY2;

        double UUJ = V[i][j];
        double CC = sqrt(GAMMAN * P[i][j] / RHO[i][j]);
        double XX = (sqrt(pow((ROEEP * CC), 2) + pow(UUJ, 2)) + CC);
    
        double PPF  = 0.5 * P[i][j];
        double UUJP = 0.5 * (UUJ + XX);
        double UUJN = 0.5 * (UUJ - XX);

        DFI[i][j][0] = (RHO[i][j]) * UUJP;
        DFI[i][j][1] = (RHOU[i][j]) * UUJP;
        DFI[i][j][2] = (RHOV[i][j]) * UUJP + PPF;
        DFI[i][j][3] = (E[i][j]) * UUJP + PPF * UUJ;

        DFJ[i][j][0] = (RHO[i][j]) * UUJN;
        DFJ[i][j][1] = (RHOU[i][j]) * UUJN;
        DFJ[i][j][2] = (RHOV[i][j]) * UUJN + PPF;
        DFJ[i][j][3] = (E[i][j]) * UUJN + PPF * UUJ;

    }    

}

__global__ void DFYI() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < ((NX - 2) * 4)) {

        unsigned long n = id / (NX - 2);
        unsigned long i = id % (NX - 2);
        // unsigned long j = id % (NY2 - 2);

        for(unsigned long j = 0 ; j < NY2 - 2 ; j++)
            DFE[i + 1][j + 1][n] = -(DFI[i + 1][j + 1][n] - DFI[i + 1][j][n]) / DY - (DFJ[i + 1][j + 2][n] - DFJ[i + 1][j + 1][n]) / DY;

    }    

    // if(!id)
    //     printf("At DFYI, DFF[1][1][0]=%.16f, DFI[1][1][0]=%.16f, DFI[1][0][0]=%.16f, DFJ[1][2][0]=%.16f, DFJ[1][1][0]=%.16f, DX=%.16f, DY=%.16f\n", DFF[1][1][0], DFI[1][1][0], DFI[1][0][0], DFJ[1][2][0], DFJ[1][1][0], DX, DY);    

}

__global__ void DUT() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 798)
    //     // printf("At DUT, DU=%.16f, DFF=%.16f, DFE=%.16f\n",DU[3][1][2], DFF[3][1][2], DFE[3][1][2]);
    //     printf("Before DUT, i=%lu j=%lu, DU=%.16f, DFF=%.16f, DFE=%.16f\n",id / (NY2 - 2), id % (NY2 - 2), DU[3][1][2], DFF[3][1][2], DFE[3][1][2]);


    if(id < ((NX - 2) * (NY2 - 2))) {

        unsigned long i = id / (NY2 - 2);
        unsigned long j = id % (NY2 - 2);

        // unsigned long i = (id / 4) / (NY2 - 2);
        // unsigned long j = (id / 4) % (NY2 - 2);
        // unsigned long n = id / 4;


        for(unsigned long n = 0 ; n < 4 ; n++)
            DU[i + 1][j + 1][n] = DFF[i + 1][j + 1][n] + DFE[i + 1][j + 1][n]; 

        // DU[i + 1][j + 1][n] = DFF[i + 1][j + 1][n] + DFE[i + 1][j + 1][n]; 

    }    

    // if(!id)
    //     printf("At DUT, DFF[1][1][0]=%.16f, DFE[1][1][0]=%.16f\n", DFF[1][1][0], DFE[1][1][0]);
    // if(id == 798)
    //     // printf("At DUT, DU=%.16f, DFF=%.16f, DFE=%.16f\n",DU[3][1][2], DFF[3][1][2], DFE[3][1][2]);
    //     printf("After DUT, i=%lu j=%lu, DU=%.16f, DFF=%.16f, DFE=%.16f\n",id / (NY2 - 2), id % (NY2 - 2), DU[3][1][2], DFF[3][1][2], DFE[3][1][2]);


}

__global__ void NEW_U() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id == 1197) {
    //     unsigned long i = id / (NY2 - 2);
    //     unsigned long j = id % (NY2 -2);
    //     printf("NEW_U before, i=%lu j=%lu, RHOV_prev=%.16f, DU=%.16f, DT=%.16f\n", i, j, RHOV[i][j], DU[i][j][2], DT);
    // }

    if(id < ((NX - 2) * (NY2 - 2))) {

        unsigned long i = id / (NY2 - 2);
        unsigned long j = id % (NY2 - 2);

        RHO[i + 1][j + 1] = RHO[i + 1][j + 1] + DU[i + 1][j + 1][0] * DT;
        RHOU[i + 1][j + 1] = RHOU[i + 1][j + 1] + DU[i + 1][j + 1][1] * DT;
        RHOV[i + 1][j + 1] = RHOV[i + 1][j + 1] + DU[i + 1][j + 1][2] * DT;              
        E[i + 1][j + 1] = E[i + 1][j + 1] + DU[i + 1][j + 1][3] * DT;

    }    

    // if(!id) {
    //     printf("Device at NEW_U for [1][0], RHO=%.16f, DU[0]=%.16f, DT=%.16f\n", RHO[1][0], DU[1][0][0], DT);
    //     printf("Device at NEW_U for [1][1], RHO=%.16f, DU[0]=%.16f, DT=%.16f\n", RHO[1][1], DU[1][1][0], DT);
    // }
    // if(id == 798) {
    //     unsigned long i = id / (NY2 - 2);
    //     unsigned long j = id % (NY2 -2);
    //     printf("NEW_U after, i=%lu j=%lu, RHOV_prev=%.16f, DU=%.16f, DT=%.16f\n", i, j, RHOV[i][j], DU[i][j][2], DT);
    // }
}

__global__ void NEW_VARIABLES() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < ((NX - 2) * (NY2 - 2))) {

        unsigned long i = id / (NY2 - 2);
        unsigned long j = id % (NY2 - 2);

        U[i + 1][j + 1] = RHOU[i + 1][j + 1] / RHO[i + 1][j + 1];
        V[i + 1][j + 1] = RHOV[i + 1][j + 1] / RHO[i + 1][j + 1];
        P[i + 1][j + 1] = (E[i + 1][j + 1] - 0.5 * RHO[i + 1][j + 1] * (pow(U[i + 1][j + 1], 2) + pow(V[i + 1][j + 1], 2))) * (GAMMAN - 1.0);

    }    

    // if(id==798)
    //     printf("New variables, %lu %lu %.16f, %.16f, %.16f, DU=%.16f, DFF=%.16f, DFE=%.16f\n", id / (NY2 - 2), id % (NY2 - 2), V[3][1], RHOV[3][1], RHO[3][1], DU[3][1][2], DFF[3][1][2], DFE[3][1][2]);

}

__global__ void timestep_preprocessing() {

    DT1 = 1.0;

}

__device__ void atomicMinDouble(double* address, double val) {

    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        // Get the minimum of val and the current value at address
        double min_val = fmin(val, __longlong_as_double(assumed));
        // Attempt to update to the new minimum value atomically
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(min_val));
    } while (assumed != old);

}

// __device__ double DDT_vector[(h_NX - 2) * (h_NY2 - 2)];

__global__ void timestep_kernel() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < ((NX - 2) * (NY2 - 2))) {

        unsigned long i = id / (NY2 - 2);
        unsigned long j = id % (NY2 - 2);

        SOS[i + 1][j + 1] = sqrt(GAMMAN * P[i + 1][j + 1] / RHO[i + 1][j + 1]);

        // serious race condition
        double DDT = CFL * DX / (fabs(U[i + 1][j + 1] + SOS[i + 1][j + 1]));
        // DT1 = fmin(DDT, DT1);
        // atomicMin(&DT1, DDT);
        // DDT_vector[i][j] = DDT;
        // DDT_vector[id] = DDT;
        atomicMinDouble(&DT1, DDT);

    }    

}

__global__ void timestep_postprocessing() {

    // DT1 = fmin(DDT, DT1);

    // DT1 = *min_elem_iter;

    DT = DT1;
    // printf("At timestep_postprocessing, DT is %.16f\n", DT);

}

void timestep() {

    timestep_preprocessing<<<1, 1>>>();

    int thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
    cudaDeviceSynchronize();

    // double *DDT_vector_pointer = thrust::raw_pointer_cast(DDT_vector.data());

    timestep_kernel<<<thread_blocks, THREADS_PER_BLOCK>>>();  

    // thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(DDT_vector);
    cudaDeviceSynchronize();
    // thrust::device_ptr<double> min_elem_iter = thrust::min_element(DDT_vector, DDT_vector + (h_NX - 2) * (h_NY2 - 2));
    // auto min_elem_iter = thrust::min_element(DDT_vector.begin(), DDT_vector.end());

    // cudaDeviceSynchronize();
    // // double min_value = *min_elem_iter;
    // // cudaMemcpyToSymbol(DT1, thrust::raw_pointer_cast(min_elem_iter), sizeof(double), cudaMemcpyDeviceToDevice);
    // cudaDeviceSynchronize();

    timestep_postprocessing<<<1, 1>>>();
    cudaDeviceSynchronize();

}

void WRITE_FILE(int iteration) {
    char FILE_NAME[50];
    FILE *file;

    // Directly assign the filename (assuming it's constant or predefined)
    // strcpy(FILE_NAME, "PRINT000001.dat");  // For example, static filename for the first iteration
    sprintf(FILE_NAME, "PRINT%06d.dat", iteration);  // Format to create filenames like PRINT000001.dat

    // Open the file for writing
    file = fopen(FILE_NAME, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Write the variable names (similar to Fortran's WRITE statement)
    fprintf(file, " VARIABLES = \"X\", \"Y\", \"U\", \"V\", \"P\", \"RHO\", \"MACH\"\n");

    // Write the zone information
    fprintf(file, "ZONE T=\"FRAME 0\", I=%d, J=%d\n", h_NX, h_NY2);

    // printf("Host, %d %d %f %f %.10f %.16f %f\n", 1, 0, h_U[1][0], h_V[1][0], h_P[1][0], h_RHO[1][0], h_MACH[1][0]);

    // Loop through the arrays and write data
    for (int J = 0; J < h_NY2; J++) {
        for (int I = 0; I < h_NX; I++) {
            // Write the indices and the values of U, V, P, RHO, and MACH
            fprintf(file, "%d %d %f %.16f %.10f %.16f %f\n", I+1, J+1, h_U[I][J], h_V[I][J], h_P[I][J], h_RHO[I][J], h_MACH[I][J]);
        }
    }

    // Close the file
    fclose(file);
}

void copy_values_from_device_to_host() {

// h_U[h_NX][h_NY2], h_V[h_NX][h_NY2], h_P[h_NX][h_NY2], h_RHO[h_NX][h_NY2], h_MACH[h_NX][h_NY2];

    cudaMemcpyFromSymbol(h_U, U, sizeof(double) * h_NX * h_NY2, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(h_V, V, sizeof(double) * h_NX * h_NY2, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(h_P, P, sizeof(double) * h_NX * h_NY2, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(h_RHO, RHO, sizeof(double) * h_NX * h_NY2, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(h_MACH, MACH, sizeof(double) * h_NX * h_NY2, 0, cudaMemcpyDeviceToHost);

}

__global__ void device_values() {

    printf("Device, %d %d %f %f %.10f %.16f %f\n", 1, 0, U[1][0], V[1][0], P[1][0], RHO[1][0], MACH[1][0]);

}

void main_process() {

    // thrust::device_vector <double> DDT_vector((h_NX - 2) * (h_NY2 - 2));
    // double *DDT_vector_pointer = thrust::raw_pointer_cast(DDT_vector.data());

    for(unsigned long i = 0 ; i < h_NTIMES ; i++) {

        // printf("\nIteration %lu\n", i + 1);

        // !___________________________MAIN PROCESS_____________________________________

        int thread_blocks = ceil(double(h_NX * h_NY2) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();
        FXI<<<thread_blocks, THREADS_PER_BLOCK>>>();         
        // cudaDeviceSynchronize();      

        // thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        thread_blocks = ceil(double(4 * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        DFXI<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        // cudaDeviceSynchronize();      


        thread_blocks = ceil(double(h_NX * h_NY2) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        FYI<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        // cudaDeviceSynchronize();      

        // thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        thread_blocks = ceil(double((h_NX - 2) * 4) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        DFYI<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        // cudaDeviceSynchronize();      

        thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        DUT<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        // cudaDeviceSynchronize();      

        thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        NEW_U<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        // cudaDeviceSynchronize();      

        //   !__________________REMAINING VARIABLE AT N+1______________________

        thread_blocks = ceil(double((h_NX - 2) * (h_NY2 - 2)) / THREADS_PER_BLOCK);
        cudaDeviceSynchronize();    
        NEW_VARIABLES<<<thread_blocks, THREADS_PER_BLOCK>>>();  
        cudaDeviceSynchronize();      

        bc();

        //   !____________________DT USING CFL CONDITION_______________________

        timestep();

        // device_values<<< 1, 1>>>();
        cudaDeviceSynchronize();

        if(!((i + 1) % 10)) {
        // if((i + 1) == 1) {
            std::cout << "Running iteration " << i + 1 << " out of " << h_NTIMES << std::endl << std::endl; 
            cudaDeviceSynchronize();
            if((i + 1) > 4995) {
                std::cout << "Writing to file" << std::endl;
                copy_values_from_device_to_host();
                cudaDeviceSynchronize();
                WRITE_FILE(i + 1);
            }
        }

        cudaDeviceSynchronize();      
        // printf("\n\n");

    }

}

__global__ void MACH_NO() {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < (NX * NY2)) {

        unsigned long i = id / NY2;
        unsigned long j = id % NY2;

        SOS[i][j] = sqrt(GAMMAN * P[i][j] / RHO[i][j]);
        MACH[i][j] = sqrt(pow(U[i][j], 2) + pow(V[i][j], 2)) / SOS[i][j];

    }    

}

int main() {

    int ITERP = 10; 

    clock_t start_time;

    // sleep(30);
    cudaDeviceSynchronize();
    start_time = clock();

    global_variable_init<<<1, 1>>>();
    cudaDeviceSynchronize();

    // std::cout << "Checkpoint" << std::endl;

    // double *block_ptr;
    // cudaMalloc((double**)&block_ptr, ((30 * h_NX * h_NY2) + (2 * h_NX * h_NY1)) * sizeof(double));
    // cudaMemset(block_ptr, 0, ((30 * h_NX * h_NY2) + (2 * h_NX * h_NY1)) * sizeof(double));

    // memory_allocation<<<1, 1>>>(block_ptr);

    geometry();
    cudaDeviceSynchronize();
    initial();
    cudaDeviceSynchronize();

    copy_values_from_device_to_host();
    cudaDeviceSynchronize();
    WRITE_FILE(0);

    bc();

    std::cout << "Checkpoint after init" << std::endl << std::endl;

    main_process();
    cudaDeviceSynchronize();

    int thread_blocks = ceil(double(h_NX * h_NY2) / THREADS_PER_BLOCK);
    MACH_NO<<<thread_blocks, THREADS_PER_BLOCK>>>();        

    // std::cout << "Checkpoint" << std::endl;

    cudaDeviceSynchronize();
    start_time = clock() - start_time;

    std::cout << "Time taken:" << double(start_time) / CLOCKS_PER_SEC << " seconds" << std::endl;


    // Host variables
    // int h_a = 5;
    // int h_b = 8;
    // int h_result;

    // // Device variable
    // int *d_result;

    // clock_t start_time, end_time;

    // start_time = clock();

    // // Allocate memory on the device for the result
    // cudaMalloc((void**)&d_result, sizeof(int));

    // // Launch the kernel with a single thread
    // add<<<1, 1>>>(h_a, h_b, d_result);

    // // Copy the result from device to host
    // cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();
    // end_time = clock() - start_time;

    // // Print the result
    // std::cout << "Result: " << h_result << std::endl;
    // std::cout << "Time taken:" << double(end_time) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // // Free device memory
    // cudaFree(d_result);

    return 0;
}