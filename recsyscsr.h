#ifndef recsyscsr_h
#define recsyscsr_h
#include <omp.h>

void matmat_transB_half(float *C, float *A, float *BT, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    for (int i = 0; i < m; i++)
        for (int j = i; j < n; j++){
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * BT[j * k + kk];
            if( j != i)
            {
                C[j * m + i] = C[i * n + j];
            }
        }
}

void matmat_transB_halfp(float *C, float *A, float *BT, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = i; j < n; j++){
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * BT[j * k + kk];
            if( j != i)
            {
                C[j * m + i] = C[i * n + j];
            }
        }
}

void matmats(float *C, float *A, float *B, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = i; j < n; j++)   //potimized to upper triangle
        {
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * B[kk * n + j];
            if( j != i)
            {
                C[j * m + i] = C[i * n + j];
            }
        }
    }
}

void matmatx(float *C, float *A, float *B, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    // #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = i; j < n; j++)   //potimized to upper triangle
        {
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * B[kk * n + j];
            if( j != i)
            {
                C[j * m + i] = C[i * n + j];
            }
        }
    }
}

// A is m x n, AT is n x m
void transposeY(float *AT, float *A, int m, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}

void matvecY(float *A, float *x, float *y, int m, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
            y[i] += A[i * n + j] * x[j];
    }
}

void updateX_recsys_csr(int *csrRowPtrR,int *csrColIdxR,float *csrValR, float *X, float *Y,
                    int m, int n, int f, float lamda,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    // struct timeval t1, t2;
    // // malloc smat (A) and svec (b)
    // float *smat = (float *)malloc(sizeof(float) * f * f);
    // float *svec = (float *)malloc(sizeof(float) * f);
#pragma omp parallel for
    for (int u = 0; u < m; u++)
    {
        struct timeval t1, t2;
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);
        gettimeofday(&t1, NULL);
        //printf("\n u = %i", u);
        float *xu = &X[u * f];

        // find nzr (i.e., #nonzeros in the uth row of R)
        int nzr = 0;
        int row_start = csrRowPtrR[u];
        int row_end = csrRowPtrR[u+1];
        nzr = row_end - row_start;
        // for (int k = 0; k < n; k++)
        //    nzr = R[u * n + k] == 0 ? nzr : nzr + 1;

        // malloc ru (i.e., uth row of R) and insert entries into it
        float *ru = (float *)malloc(sizeof(float) * nzr);
        
// gettimeofday(&t2, NULL);
// *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
// gettimeofday(&t1, NULL);
        
        // int count = 0;
        // #pragma omp parallel for
        for(int k = row_start; k < row_end; k++)
        // for(int k = csrRowPtrR[u]; k < csrRowPtrR[u+1]; k++)
        {
            ru[k-row_start] = csrValR[k];
            // count++;
        }
// gettimeofday(&t2, NULL);
// *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
// gettimeofday(&t1, NULL);

        // for (int k = 0; k < n; k++)
        // {
        //     if (R[u * n + k] != 0)
        //     {
        //         ru[count] = R[u * n + k];
        //         count++;
        //     }
        // }
        //printf("\n nzr = %i, ru = \n", nzr);
        //printvec(ru, nzr);

        // create sY and sYT (i.e., the zero-free version of Y and YT)
        float *sY = (float *)malloc(sizeof(float) * nzr * f);
        float *sYT = (float *)malloc(sizeof(float) * nzr * f);
        // fill sY, according to the sparsity of the uth row of R

        // count = 0;
        // #pragma omp parallel for
        for (int k = row_start; k < row_end; k++)
        {
            memcpy(&sY[(k-row_start) * f], &Y[csrColIdxR[k] * f], sizeof(float) * f);
            // count++;
        }
// gettimeofday(&t2, NULL);
// *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
// gettimeofday(&t1, NULL);
        
        // for (int k = 0; k < n; k++)
        // {
        //     if (R[u * n + k] != 0)
        //     {
        //         memcpy(&sY[count * f], &Y[k * f], sizeof(float) * f);
        //         count++;
        //     }
        // }
        //printf("\n sY = \n");
        //printmat(sY, nzr, f);

        // transpose sY to sYT
        transpose(sYT, sY, nzr, f);

        // multiply sYT and sY, and plus lamda * I
        matmat_transB_half(smat, sYT, sYT, f, nzr, f);
        for (int i = 0; i < f; i++)
            smat[i * f + i] += lamda;

        gettimeofday(&t2, NULL);
        // *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // compute b (i.e., svec) by multiplying sYT and the uth row of R
        gettimeofday(&t1, NULL);
        matvec(sYT, ru, svec, f, nzr);
        gettimeofday(&t2, NULL);
        // *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // solve the system of Ax=b, and get x = the uth row of X
        gettimeofday(&t1, NULL);
        int cgiter = 0;
        cg(smat, xu, svec, f, &cgiter, 100, 0.00001);
        gettimeofday(&t2, NULL);
        // *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nxu = \n");
        //printvec(xu, f);
        
        free(ru);
        free(sY);
        free(sYT);

        free(smat);
        free(svec);
    }

    // free(smat);
    // free(svec);
    //free(YT);
}

void updateY_recsys_csr(int *csrRowPtrR,int *csrColIdxR,float *csrValR, float *X, float *Y,
                    int m, int n, int f, float lamda,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    // struct timeval t1, t2;

    // float *smat = (float *)malloc(sizeof(float) * f * f);
    // float *svec = (float *)malloc(sizeof(float) * f);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        //printf("\n i = %i", i);
        float *yi = &Y[i * f];

        int nzc = 0;
        for(int k = 0; k < csrRowPtrR[m]; k++)
        {
            if(csrColIdxR[k]==i)
            {
               nzc += 1; 
            }
        }
        // for (int k = 0; k < m; k++)
        //     nzc = R[k * n + i] == 0 ? nzc : nzc + 1;

        float *ri = (float *)malloc(sizeof(float) * nzc);
        int count = 0;
        for(int k = 0; k < csrRowPtrR[m]; k++)
        {
            if(csrColIdxR[k]==i)
            {
               ri[count] = csrValR[k];
               count++;
            }
        }
        // for (int k = 0; k < m; k++)
        // {
        //     if (R[k * n + i] != 0)
        //     {
        //         ri[count] = R[k * n + i];
        //         count++;
        //     }
        // }
        //printf("\n nzc = %i, ri = \n", nzc);
        //printvec(ri, nzc);

        float *sX = (float *)malloc(sizeof(float) * nzc * f);
        float *sXT = (float *)malloc(sizeof(float) * nzc * f);
        count = 0;
        for(int k = 0; k < csrRowPtrR[m]; k++)
        {
            if(csrColIdxR[k]==i)
            {
               int z;
               for(z = 1; z < m+1; z++)
               {
                   if(k < csrRowPtrR[z])
                        break;
               }
               memcpy(&sX[count * f], &X[(z-1) * f], sizeof(float) * f);
               count++;
            }
        }
        transpose(sXT, sX, nzc, f);
        // for (int k = 0; k < m; k++)
        // {
        //     if (R[k * n + i] != 0)
        //     {
        //         memcpy(&sX[count * f], &X[k * f], sizeof(float) * f);
        //         count++;
        //     }
        // }
        //printf("\n sX = \n");
        //printmat(sX, nzc, f);
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        matmats(smat, sXT, sX, f, nzc, f);
        for (int i = 0; i < f; i++)
            smat[i * f + i] += lamda;

        gettimeofday(&t2, NULL);
        // *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        matvec(sXT, ri, svec, f, nzc);
        gettimeofday(&t2, NULL);
        // *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        int cgiter = 0;
        cg(smat, yi, svec, f, &cgiter, 100, 0.00001);
        gettimeofday(&t2, NULL);
        // *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nyi = \n");
        //printvec(yi, f);

        free(ri);
        free(sX);
        free(sXT);

        free(smat);
        free(svec);
    }

    // free(smat);
    // free(svec);
}
void printvecint(int *cscColPtrR,int n);

void updateY_recsys_csc(int *cscColPtrR,int *cscRowIdxR,float *cscValR, float *X, float *Y,
                    int m, int n, int f, float lamda,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    // struct timeval t1, t2;

    // float *smat = (float *)malloc(sizeof(float) * f * f);
    // float *svec = (float *)malloc(sizeof(float) * f);
// #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        //printf("\n i = %i", i);
        float *yi = &Y[i * f];

        int nzc = 0;
        int col_start = cscColPtrR[i];
        int col_end = cscColPtrR[i+1];
        nzc = col_end - col_start;
        
        // for (int k = 0; k < m; k++)
        //     nzc = R[k * n + i] == 0 ? nzc : nzc + 1;

        float *ri = (float *)malloc(sizeof(float) * nzc);
        // int count = 0;
        #pragma omp parallel for
        for (int k = col_start; k < col_end; k++)
        {
            ri[k-col_start] = cscValR[k];
            // count++;
        }
        
        // for (int k = 0; k < m; k++)
        // {
        //     if (R[k * n + i] != 0)
        //     {
        //         ri[count] = R[k * n + i];
        //         count++;
        //     }
        // }
        // printf("\n nzc = %i, ri = \n", nzc);
        // printvec(ri, nzc);

        float *sX = (float *)malloc(sizeof(float) * nzc * f);
        float *sXT = (float *)malloc(sizeof(float) * nzc * f);
        // count = 0;
        #pragma omp parallel for
        for (int k = col_start; k < col_end; k++)
        {
            memcpy(&sX[(k-col_start) * f], &X[cscRowIdxR[k] * f], sizeof(float) * f);
            // count++;
        }
        
        transposeY(sXT, sX, nzc, f);
        // for (int k = 0; k < m; k++)
        // {
        //     if (R[k * n + i] != 0)
        //     {
        //         memcpy(&sX[count * f], &X[k * f], sizeof(float) * f);
        //         count++;
        //     }
        // }
        // printf("\n sX = \n");
        //printmat(sX, nzc, f);
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        matmat_transB_halfp(smat, sXT, sXT, f, nzc, f);
        for (int k = 0; k < f; k++)
            smat[k * f + k] += lamda;

        gettimeofday(&t2, NULL);
        // *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        matvecY(sXT, ri, svec, f, nzc);
        gettimeofday(&t2, NULL);
        // *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        int cgiter = 0;
        cg(smat, yi, svec, f, &cgiter, 100, 0.00001);
        gettimeofday(&t2, NULL);
        // *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nyi = \n");
        //printvec(yi, f);

        free(ri);
        free(sX);
        free(sXT);

        free(smat);
        free(svec);
    }

    // free(smat);
    // free(svec);
}

#include "tranpose.h"
void printvecint(int *x, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d:%d\n",i, x[i]);
}
void als_recsys_csr(int *csrRowPtrR,int *csrColIdxR,float *csrValR, float *X, float *Y,
         int m, int n, int f, float lamda)
{
    int nnzR = csrRowPtrR[m];
    // create YT and Rp
    float *YT = (float *)malloc(sizeof(float) * f * n);
    //float *Rp = (float *)malloc(sizeof(float) * m * n);

    int iter = 0;
    float error = 0.0;
    float error_old = 0.0;
    float error_new = 0.0;
    struct timeval t1, t2;

    double time_updatex_prepareA = 0;
    double time_updatex_prepareb = 0;
    double time_updatex_solver = 0;

    double time_updatey_prepareA = 0;
    double time_updatey_prepareb = 0;
    double time_updatey_solver = 0;

    double time_updatex = 0;
    double time_updatey = 0;
    double time_validate = 0;

    int *cscRowIdxR = (int *)malloc(sizeof(int) * nnzR);
    int *cscColPtrR = (int *)malloc(sizeof(int) * (n+1));
    float *cscValR = (float *)malloc(sizeof(float) * nnzR);
    matrix_transposition(m, n, nnzR, csrRowPtrR, csrColIdxR, csrValR, cscRowIdxR, cscColPtrR, cscValR);
    // printvecint(cscColPtrR,(n+1));
    do
    {
        // printf("the process of updateX\n");
        // step 1. update X
        gettimeofday(&t1, NULL);
        updateX_recsys_csr(csrRowPtrR,csrColIdxR,csrValR, X, Y, m, n, f, lamda,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // printf("the process of updateY\n");
        // step 2. update Y
        gettimeofday(&t1, NULL);
        // updateY_recsys_csr(csrRowPtrR,csrColIdxR,csrValR, X, Y, m, n, f, lamda,
        //                &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        
        // updateX_recsys_csr(cscColPtrR,cscRowIdxR,cscValR, X, Y, n, m, f, lamda,
        //                &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        updateY_recsys_csc(cscColPtrR,cscRowIdxR,cscValR, X, Y, m, n, f, lamda,
                       &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        
        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 3. validate
        // step 3-1. matrix multiplication
        gettimeofday(&t1, NULL);
        // matmat_transB(Rp, X, Y, m, f, n);

        // step 3-2. calculate error
        error_new = 0.0;
        int nnz = 0;
        int row = 0;
        for (int i = 0; i < nnzR; i++)
        {
            float resultXY = 0;
            int col = csrColIdxR[i];
            if (i >= csrRowPtrR[row+1])
            {
                row++;
            }
            
            for (int index = 0; index < f; index++)
            {
                resultXY += X[row * f + index] * Y[col * f + index];
            }
            // printf("%f %f %f %f\n",error_new,fabs(csrValR[i] - resultXY),csrValR[i],resultXY);
            error_new += (csrValR[i] - resultXY) * (csrValR[i] - resultXY);
            // printf("%f\n",error_new);
        }
        
        // for (int i = 0; i < m * n; i++)
        // {
        //     if (R[i] != 0)
        //     {
        //         error_new += fabs(Rp[i] - R[i]) * fabs(Rp[i] - R[i]);
        //         nnz++;
        //     }
        // }
        error_new = sqrt(error_new/nnzR);
        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        error = fabs(error_new - error_old) / error_new;
        error_old = error_new;
        printf("iter = %i, error = %f\n", iter, error);

        iter++;
    }
    while(iter < 1000 && error > 0.0001);
    free(cscColPtrR);
    free(cscRowIdxR);
    free(cscValR);
    //printf("\nR = \n");
    //printmat(R, m, n);

    //printf("\nRp = \n");
    //printmat(Rp, m, n);
    printf("\nfunction : csr\n");
    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);
    printf("The sum of time is : %f ms\n",time_updatex+time_updatey+time_validate);
    // free(Rp);
    free(YT);
}

#endif