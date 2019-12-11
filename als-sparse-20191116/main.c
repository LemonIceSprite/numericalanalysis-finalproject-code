#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <omp.h>

#include <immintrin.h>

#include "mmio_highlevel.h"
#include "tranpose.h"

void printmat(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%4.2f ", A[i * n + j]);
        printf("\n");
    }
}

void printvec(float *x, int n)
{
    for (int i = 0; i < n; i++)
        printf("%4.2f\n", x[i]);
}



void matmat(float *C, float *A, float *B, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * B[kk * n + j];
}

void matmat_transB(float *C, float *A, float *BT, int m, int k, int n)
{
    memset(C, 0, sizeof(float) * m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int kk = 0; kk < k; kk++)
                C[i * n + j] += A[i * k + kk] * BT[j * k + kk];
}

float dotproduct(float *vec1, float *vec2, int n)
{
    float result = 0;
    for (int i = 0; i < n; i++)
        result += vec1[i] * vec2[i];
    return result;
}

float dotproduct_avx512(float *vec1, float *vec2, int n)
{
    float result = 0;
    //for (int i = 0; i < n; i++)
    //    result += vec1[i] * vec2[i];
    
    __m512 r0,c0,d1;

    d1 = _mm512_setzero_ps();

    r0 = _mm512_loadu_ps(vec1);
    c0 = _mm512_loadu_ps(vec2);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                   

    r0 = _mm512_loadu_ps(&vec1[16]);
    c0 = _mm512_loadu_ps(&vec2[16]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                    

    r0 = _mm512_loadu_ps(&vec1[32]);
    c0 = _mm512_loadu_ps(&vec2[32]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                    

    r0 = _mm512_loadu_ps(&vec1[48]);
    c0 = _mm512_loadu_ps(&vec2[48]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                   
    result = _mm512_reduce_add_ps(d1);

    return result;
}

float dotproduct_avx512_reg(float *vec1, __m512 c0, __m512 c1, __m512 c2, __m512 c3, int n)
{
    float result = 0;
    //for (int i = 0; i < n; i++)
    //    result += vec1[i] * vec2[i];
    
    __m512 r0, d1;

    d1 = _mm512_setzero_ps();

    r0 = _mm512_loadu_ps(vec1);
    //c0 = _mm512_loadu_ps(vec2);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                   

    r0 = _mm512_loadu_ps(&vec1[16]);
    //c0 = _mm512_loadu_ps(&vec2[16]);
    d1 = _mm512_fmadd_ps(r0, c1, d1);
                    

    r0 = _mm512_loadu_ps(&vec1[32]);
    //c0 = _mm512_loadu_ps(&vec2[32]);
    d1 = _mm512_fmadd_ps(r0, c2, d1);
                    

    r0 = _mm512_loadu_ps(&vec1[48]);
    //c0 = _mm512_loadu_ps(&vec2[48]);
    d1 = _mm512_fmadd_ps(r0, c3, d1);
                   
    result = _mm512_reduce_add_ps(d1);

    return result;
}

void matvec(float *A, float *x, float *y, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
            y[i] += A[i * n + j] * x[j];
    }
}

void matvec_avx512(float *A, float *x, float *y, int m, int n)
{
    __m512 c0 = _mm512_loadu_ps(x);
    __m512 c1 = _mm512_loadu_ps(&x[16]);
    __m512 c2 = _mm512_loadu_ps(&x[32]);
    __m512 c3 = _mm512_loadu_ps(&x[48]);

    #pragma GCC unroll 64
    for (int i = 0; i < 64; i++)
    {
        //y[i] = 0;
        //for (int j = 0; j < n; j++)
            //y[i] += A[i * n + j] * x[j];

        y[i] = dotproduct_avx512_reg(&A[i * n], c0, c1, c2, c3, n);
    }
}

// A is m x n, AT is n x m
void transpose(float *AT, float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}

float vec2norm(float *x, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

float vec2norm_avx512(float *x, int n)
{
    float sum = 0;
    //for (int i = 0; i < n; i++)
        //sum += x[i] * x[i];
    __m512 r0, d1;
                    
    d1 = _mm512_setzero_ps ();
    r0 = _mm512_loadu_ps(x);
    //c0 = _mm512_loadu_ps(vec2);
    d1 = _mm512_fmadd_ps (r0, r0, d1);
                   

    r0 = _mm512_loadu_ps(&x[16]);
    //c0 = _mm512_loadu_ps(&vec2[16]);
    d1 = _mm512_fmadd_ps (r0, r0, d1);
                    

    r0 = _mm512_loadu_ps(&x[32]);
    //c0 = _mm512_loadu_ps(&vec2[32]);
    d1 = _mm512_fmadd_ps (r0, r0, d1);
                    

    r0 = _mm512_loadu_ps(&x[48]);
    //c0 = _mm512_loadu_ps(&vec2[48]);
    d1 = _mm512_fmadd_ps (r0, r0, d1);
                   
    sum = _mm512_reduce_add_ps (d1);


    return sqrt(sum);
}

void cholesky(float *A, float *x, float *b, int n)
{
    float *At = (float *)malloc(n * n * sizeof(float));
    memcpy(At, A, n * n * sizeof(float));
    float *bt = (float *)malloc(n * sizeof(float));
    memcpy(bt, b, n * sizeof(float));
    memset(x, 0, n * sizeof(float));

    float *L = (float *)malloc(n * n * sizeof(float));
    memset(L, 0, n * n * sizeof(float));

    // Cholesky decomposition
    for (int i = 0; i < n; i++)
    {
        // diag
        float sum = 0;
        for (int j = 0; j < i; j++)
            sum += L[i * n + j] * L[i * n + j];
        L[i * n + i] = sqrt(At[i * n + i] - sum);

        // non-diag
        for (int k = i + 1; k < n; k++)
        {
            float sum = 0;
            for (int s = 0; s < i; s++)
                sum += L[k * n + s] * L[i * n + s]; //L[s * n + i];
            L[k * n + i] = (At[k * n + i] - sum) / L[i * n + i];
        }
    }

    //printf("\nL = \n");
    //printmat(L, n);

    // forward substitution
    float *y = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        float sum = 0;
        for (int j = 0; j < i; j++)
            sum += L[i * n + j] * y[j];
        y[i] = (bt[i] - sum) / L[i * n + i];
    }

    //printf("\ny = \n");
    //printvec(y, n);

    // backward substitution
    for (int i = n - 1; i >= 0; i--)
    {
        float sum = 0;
        for (int j = i + 1; j < n; j++)
            sum += L[j * n + i] * x[j];
        x[i] = (y[i] - sum) / L[i * n + i];
    }

    //printf("\nx = \n");
    //printvec(x, n);

    free(y);
    free(L);
    free(At);
    free(bt);
}

void cg_opt(float *A, float *x, float *b,
        float *residual, float *y, float *p, float *q,
        int n, int *iter, int maxiter, float threshold)
{
    memset(x, 0, sizeof(float) * n);
    *iter = 0;
    float norm = 0;
    float rho = 0;
    float rho_1 = 0;

    // p0 = r0 = b - Ax0
    matvec(A, x, y, n, n);
    for (int i = 0; i < n; i++)
        residual[i] = b[i] - y[i];
    //printvec(residual, n);

    do
    {
        //printf("\ncg iter = %i\n", *iter);
        rho = dotproduct(residual, residual, n);
        //printf("rho = %f\n",rho);
        if (*iter == 0)
        {
            for (int i = 0; i < n; i++)
                p[i] = residual[i];
        }
        else
        {
            float beta = rho / rho_1;
            for (int i = 0; i < n; i++)
                p[i] = residual[i] + beta * p[i];
        }

        matvec(A, p, q, n, n);
        float alpha = rho / dotproduct(p, q, n);
        //printf("rho = %f\n",rho);
        //printf("alpha = %f\n", alpha);
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        for (int i = 0; i < n; i++)
            residual[i] += - alpha * q[i];

        rho_1 = rho;
        float error = vec2norm(residual, n) / vec2norm(b, n);

        //printvec(x, n);
        *iter += 1;

        if (error < threshold)
            break;
    }
    while (*iter < maxiter);
}

void cg_opt_avx512(float *A, float *x, float *b,
        float *residual, float *y, float *p, float *q,
        int n, int *iter, int maxiter, float threshold)
{
    memset(x, 0, sizeof(float) * n);
    *iter = 0;
    float norm = 0;
    float rho = 0;
    float rho_1 = 0;

    // p0 = r0 = b - Ax0
    matvec_avx512(A, x, y, n, n);
    for (int i = 0; i < n; i++)
        residual[i] = b[i] - y[i];
    //printvec(residual, n);

    do
    {
        //printf("\ncg iter = %i\n", *iter);
        rho = dotproduct_avx512(residual, residual, n);
        //printf("rho = %f\n",rho);
        if (*iter == 0)
        {
            for (int i = 0; i < n; i++)
                p[i] = residual[i];
        }
        else
        {
            float beta = rho / rho_1;
            for (int i = 0; i < n; i++)
                p[i] = residual[i] + beta * p[i];
        }

        matvec_avx512(A, p, q, n, n);
        float alpha = rho / dotproduct_avx512(p, q, n);
        //printf("alpha = %f\n",rho);
        //printf("alpha = %f\n", alpha);
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        for (int i = 0; i < n; i++)
            residual[i] += - alpha * q[i];

        rho_1 = rho;
        float error = vec2norm_avx512(residual, n) / vec2norm_avx512(b, n);

        //printvec(x, n);
        *iter += 1;

        if (error < threshold)
            break;
    }
    while (*iter < maxiter);
}

void cg(float *A, float *x, float *b, int n, int *iter, int maxiter, float threshold)
{
    memset(x, 0, sizeof(float) * n);
    float *residual = (float *)malloc(sizeof(float) * n);
    float *y = (float *)malloc(sizeof(float) * n);
    float *p = (float *)malloc(sizeof(float) * n);
    float *q = (float *)malloc(sizeof(float) * n);
    *iter = 0;
    float norm = 0;
    float rho = 0;
    float rho_1 = 0;

    // p0 = r0 = b - Ax0
    matvec(A, x, y, n, n);
    for (int i = 0; i < n; i++)
        residual[i] = b[i] - y[i];
    //printvec(residual, n);

    do
    {
        //printf("\ncg iter = %i\n", *iter);
        rho = dotproduct(residual, residual, n);
        if (*iter == 0)
        {
            for (int i = 0; i < n; i++)
                p[i] = residual[i];
        }
        else
        {
            float beta = rho / rho_1;
            for (int i = 0; i < n; i++)
                p[i] = residual[i] + beta * p[i];
        }

        matvec(A, p, q, n, n);
        float alpha = rho / dotproduct(p, q, n);
        //printf("alpha = %f\n", alpha);
        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        for (int i = 0; i < n; i++)
            residual[i] += - alpha * q[i];

        rho_1 = rho;
        float error = vec2norm(residual, n) / vec2norm(b, n);

        //printvec(x, n);
        *iter += 1;

        if (error < threshold)
            break;
    }
    while (*iter < maxiter);

    free(residual);
    free(y);
    free(p);
    free(q);
}

void updateX(float *R, float *X, float *Y,
             int m, int n, int f, float lamda)
{
    // create YT, and transpose Y
    float *YT = (float *)malloc(sizeof(float) * n * f);
    transpose(YT, Y, n, f);

    // create smat = YT*Y
    float *smat = (float *)malloc(sizeof(float) * f * f);

    // multiply YT and Y to smat
    matmat(smat, YT, Y, f, n, f);

    // smat plus lamda*I
    for (int i = 0; i < f; i++)
        smat[i * f + i] += lamda;

    // create svec
    float *svec = (float *)malloc(sizeof(float) * f);

    // loop for all rows of X, from 0 to m-1
    for (int u = 0; u < m; u++)
    {
        printf("\n u = %i", u);
        // reference the uth row of X
        float *xu = &X[u * f];

        // compute svec by multiplying YT and the uth row of R
        matvec(YT, &R[u * n], svec, f, n);

        // solve the Ax=b system (A is smat, b is svec, x is xu (uth row of X))
        int cgiter = 0;
        cg(smat, xu, svec, f, &cgiter, 100, 0.00001);

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nxu = \n");
        //printvec(xu, f);
    }

    free(smat);
    free(svec);
    free(YT);
}

void updateY(float *R, float *X, float *Y,
             int m, int n, int f, float lamda)
{
    float *XT = (float *)malloc(sizeof(float) * m * f);
    transpose(XT, X, m, f);

    float *smat = (float *)malloc(sizeof(float) * f * f);
    matmat(smat, XT, X, f, m, f);
    for (int i = 0; i < f; i++)
        smat[i * f + i] += lamda;

    float *svec = (float *)malloc(sizeof(float) * f);
    float *ri = (float *)malloc(sizeof(float) * m);

    for (int i = 0; i < n; i++)
    {
        printf("\n i = %i", i);
        float *yi = &Y[i * f];

        for (int k = 0; k < m; k++)
            ri[k] = R[k * n + i];

        matvec(XT, ri, svec, f, m);

        int cgiter = 0;
        cg(smat, yi, svec, f, &cgiter, 100, 0.00001);

        printf("\nsmat = \n");
        printmat(smat, f, f);

        printf("\nsvec = \n");
        printvec(svec, f);

        printf("\nyi = \n");
        printvec(yi, f);
    }

    free(smat);
    free(svec);
    free(ri);
    free(XT);
}

// ALS for matrix factorization
void als(float *R, float *X, float *Y,
         int m, int n, int f, float lamda)
{
    // create YT
    float *YT = (float *)malloc(sizeof(float) * f * n);

    // create R (result)
    float *Rp = (float *)malloc(sizeof(float) * m * n);

    int iter = 0;
    float error = 0.0;
    float error_old = 0.0;
    float error_new = 0.0;
    do
    {
        // step 1. update X
        updateX(R, X, Y, m, n, f, lamda);

        // step 2. update Y
        updateY(R, X, Y, m, n, f, lamda);

        // step 3. validate R, by multiplying X and YT

        // step 3-1. matrix multiplication with transposed Y
        matmat_transB(Rp, X, Y, m, f, n);

        // step 3-2. calculate error
        error_new = 0.0;
        for (int i = 0; i < m * n; i++)
            if (R[i] != 0) // only compare nonzero entries in R
                error_new += fabs(Rp[i] - R[i]) * fabs(Rp[i] - R[i]);
        error_new = sqrt(error_new/(m * n));

        error = fabs(error_new - error_old) / error_new;
        error_old = error_new;
        printf("iter = %i, error = %f\n", iter, error);

        iter++;
    }
    while(iter < 10 && error > 0.0001);

    printf("\nR = \n");
    printmat(R, m, n);

    printf("\nRp = \n");
    printmat(Rp, m, n);

    free(Rp);
    free(YT);
}

void updateX_recsys(float *R, float *X, float *Y,
                    int m, int n, int f, float lamda,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    struct timeval t1, t2;

    // malloc smat (A) and svec (b)
    float *smat = (float *)malloc(sizeof(float) * f * f);
    float *svec = (float *)malloc(sizeof(float) * f);

    for (int u = 0; u < m; u++)
    {
        gettimeofday(&t1, NULL);
        //printf("\n u = %i", u);
        float *xu = &X[u * f];

        // find nzr (i.e., #nonzeros in the uth row of R)
        int nzr = 0;
        for (int k = 0; k < n; k++)
            nzr = R[u * n + k] == 0 ? nzr : nzr + 1;

        // malloc ru (i.e., uth row of R) and insert entries into it
        float *ru = (float *)malloc(sizeof(float) * nzr);
        int count = 0;
        for (int k = 0; k < n; k++)
        {
            if (R[u * n + k] != 0)
            {
                ru[count] = R[u * n + k];
                count++;
            }
        }
        //printf("\n nzr = %i, ru = \n", nzr);
        //printvec(ru, nzr);

        // create sY and sYT (i.e., the zero-free version of Y and YT)
        float *sY = (float *)malloc(sizeof(float) * nzr * f);
        float *sYT = (float *)malloc(sizeof(float) * nzr * f);

        // fill sY, according to the sparsity of the uth row of R
        count = 0;
        for (int k = 0; k < n; k++)
        {
            if (R[u * n + k] != 0)
            {
                memcpy(&sY[count * f], &Y[k * f], sizeof(float) * f);
                count++;
            }
        }
        //printf("\n sY = \n");
        //printmat(sY, nzr, f);

        // transpose sY to sYT
        transpose(sYT, sY, nzr, f);

        // multiply sYT and sY, and plus lamda * I
        matmat(smat, sYT, sY, f, nzr, f);
        for (int i = 0; i < f; i++)
            smat[i * f + i] += lamda;

        gettimeofday(&t2, NULL);
        *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // compute b (i.e., svec) by multiplying sYT and the uth row of R
        gettimeofday(&t1, NULL);
        matvec(sYT, ru, svec, f, nzr);
        gettimeofday(&t2, NULL);
        *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // solve the system of Ax=b, and get x = the uth row of X
        gettimeofday(&t1, NULL);
        int cgiter = 0;
        cg(smat, xu, svec, f, &cgiter, 100, 0.00001);
        gettimeofday(&t2, NULL);
        *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nxu = \n");
        //printvec(xu, f);

        free(ru);
        free(sY);
        free(sYT);
    }

    free(smat);
    free(svec);
    //free(YT);
}

void updateY_recsys(float *R, float *X, float *Y,
                    int m, int n, int f, float lamda,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    struct timeval t1, t2;

    float *smat = (float *)malloc(sizeof(float) * f * f);
    float *svec = (float *)malloc(sizeof(float) * f);

    for (int i = 0; i < n; i++)
    {
        gettimeofday(&t1, NULL);
        //printf("\n i = %i", i);
        float *yi = &Y[i * f];

        int nzc = 0;
        for (int k = 0; k < m; k++)
            nzc = R[k * n + i] == 0 ? nzc : nzc + 1;

        float *ri = (float *)malloc(sizeof(float) * nzc);
        int count = 0;
        for (int k = 0; k < m; k++)
        {
            if (R[k * n + i] != 0)
            {
                ri[count] = R[k * n + i];
                count++;
            }
        }
        //printf("\n nzc = %i, ri = \n", nzc);
        //printvec(ri, nzc);

        float *sX = (float *)malloc(sizeof(float) * nzc * f);
        float *sXT = (float *)malloc(sizeof(float) * nzc * f);
        count = 0;
        for (int k = 0; k < m; k++)
        {
            if (R[k * n + i] != 0)
            {
                memcpy(&sX[count * f], &X[k * f], sizeof(float) * f);
                count++;
            }
        }
        //printf("\n sX = \n");
        //printmat(sX, nzc, f);

        transpose(sXT, sX, nzc, f);
        matmat(smat, sXT, sX, f, nzc, f);
        for (int i = 0; i < f; i++)
            smat[i * f + i] += lamda;

        gettimeofday(&t2, NULL);
        *time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        matvec(sXT, ri, svec, f, nzc);
        gettimeofday(&t2, NULL);
        *time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        gettimeofday(&t1, NULL);
        int cgiter = 0;
        cg(smat, yi, svec, f, &cgiter, 100, 0.00001);
        gettimeofday(&t2, NULL);
        *time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        //printf("\nsmat = \n");
        //printmat(smat, f, f);

        //printf("\nsvec = \n");
        //printvec(svec, f);

        //printf("\nyi = \n");
        //printvec(yi, f);

        free(ri);
        free(sX);
        free(sXT);
    }

    free(smat);
    free(svec);
}


void als_recsys(float *R, float *X, float *Y,
         int m, int n, int f, float lamda)
{
    // create YT and Rp
    float *YT = (float *)malloc(sizeof(float) * f * n);
    float *Rp = (float *)malloc(sizeof(float) * m * n);

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

    do
    {
        // step 1. update X
        gettimeofday(&t1, NULL);
        updateX_recsys(R, X, Y, m, n, f, lamda,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 2. update Y
        gettimeofday(&t1, NULL);
        updateY_recsys(R, X, Y, m, n, f, lamda,
                       &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 3. validate
        // step 3-1. matrix multiplication
        gettimeofday(&t1, NULL);
        matmat_transB(Rp, X, Y, m, f, n);

        // step 3-2. calculate error
        error_new = 0.0;
        int nnz = 0;
        for (int i = 0; i < m * n; i++)
        {
            if (R[i] != 0)
            {
                error_new += fabs(Rp[i] - R[i]) * fabs(Rp[i] - R[i]);
                nnz++;
            }
        }
        error_new = sqrt(error_new/nnz);
        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        error = fabs(error_new - error_old) / error_new;
        error_old = error_new;
        printf("iter = %i, error = %f\n", iter, error);

        iter++;
    }
    while(iter < 1000 && error > 0.0001);

    //printf("\nR = \n");
    //printmat(R, m, n);

    //printf("\nRp = \n");
    //printmat(Rp, m, n);

    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);

    free(Rp);
    free(YT);
}

void updateX_recsys_sparse(int *csrRowPtrR, int *csrColIdxR, float *csrValR, int nnz, float *X, float *Y,
                    int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    float *residual_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *y_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *p_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *q_omp = (float *)malloc(sizeof(float) * f * nthreads);

    //struct timeval t1, t2;

    // malloc smat (A) and svec (b)
    float *smat_omp = (float *)malloc(sizeof(float) * f * f * nthreads);
    float *svec_omp = (float *)malloc(sizeof(float) * f * nthreads);

    #pragma omp parallel for
    for (int u = 0; u < m; u++)
    {
        int tid = omp_get_thread_num();
        float *smat = &smat_omp[tid * f * f];
        float *svec = &svec_omp[tid * f];

        float *residual = &residual_omp[tid * f];
        float *y = &y_omp[tid * f];
        float *p = &p_omp[tid * f];
        float *q = &q_omp[tid * f];

        //gettimeofday(&t1, NULL);
        //printf("\n u = %i", u);
        float *xu = &X[u * f];

        memset(smat, 0, sizeof(float) * f * f);
        for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
        {
            float *yn = &Y[csrColIdxR[j] * f];
            for (int si = 0; si < f; si++)
            {
                float val = yn[si];
                for (int k = si; k < f; k++)
                {
                    smat[si * f + k] += val * yn[k];
                }
            }
        }
        for (int si = 0; si < f; si++)
        {
            smat[si * f + si] += lamda;
            for (int sj = si+1; sj < f; sj++)
            {
                smat[sj * f + si] = smat[si * f + sj];
            }
        }

        //gettimeofday(&t2, NULL);
        //*time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // compute b (i.e., svec) by multiplying sYT and the uth row of R
        //gettimeofday(&t1, NULL);
        //matvec(sYT, ru, svec, f, nzr);
        memset(svec, 0, sizeof(float) * f);
        for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
        {
            float *yn = &Y[csrColIdxR[j] * f];
            float val = csrValR[j];
            for (int si = 0; si < f; si++)
                svec[si] += val * yn[si];
        }
        //gettimeofday(&t2, NULL);
        //*time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // solve the system of Ax=b, and get x = the uth row of X
        //gettimeofday(&t1, NULL);
        int cgiter = 0;
//printf("\nsmat =\n");
//printmat(smat, f, f);
//printf("\nsvec =\n");
//printvec(svec, f);
        //cg(smat, xu, svec, f, &cgiter, 100, 0.00001);
        cg_opt(smat, xu, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        //cholesky(smat, xu, svec, f);
//printf("\nxu =\n");
//printvec(xu, f);
//return;
        //gettimeofday(&t2, NULL);
        //printf("u = %i, cgiter = %i\n", u, cgiter);
        //*time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    free(smat_omp);
    free(svec_omp);
    //free(YT);
    free(residual_omp);
    free(y_omp);
    free(p_omp);
    free(q_omp);
}

void updateY_recsys_sparse(int *cscColPtrR, int *cscRowIdxR, float *cscValR, int nnz, float *X, float *Y,
                    int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    float *residual_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *y_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *p_omp = (float *)malloc(sizeof(float) * f * nthreads);
    float *q_omp = (float *)malloc(sizeof(float) * f * nthreads);

    //struct timeval t1, t2;

    // malloc smat (A) and svec (b)
    float *smat_omp = (float *)malloc(sizeof(float) * f * f * nthreads);
    float *svec_omp = (float *)malloc(sizeof(float) * f * nthreads);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int tid = omp_get_thread_num();
        float *smat = &smat_omp[tid * f * f];
        float *svec = &svec_omp[tid * f];

        float *residual = &residual_omp[tid * f];
        float *y = &y_omp[tid * f];
        float *p = &p_omp[tid * f];
        float *q = &q_omp[tid * f];

        //gettimeofday(&t1, NULL);
        //printf("\n u = %i", u);
        float *yi = &Y[i * f];

        memset(smat, 0, sizeof(float) * f * f);
        for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
        {
            float *xn = &X[cscRowIdxR[j] * f];
            for (int si = 0; si < f; si++)
            {
                float val = xn[si];
                for (int k = si; k < f; k++)
                {
                    smat[si * f + k] += val * xn[k];
                }
            }
        }
        for (int si = 0; si < f; si++)
        {
            smat[si * f + si] += lamda;
            for (int sj = si+1; sj < f; sj++)
            {
                smat[sj * f + si] = smat[si * f + sj];
            }
        }

        //gettimeofday(&t2, NULL);
        //*time_prepareA += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // compute b (i.e., svec) by multiplying sYT and the uth row of R
        //gettimeofday(&t1, NULL);
        //matvec(sYT, ru, svec, f, nzr);
        memset(svec, 0, sizeof(float) * f);
        for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
        {
            float *xn = &X[cscRowIdxR[j] * f];
            float val = cscValR[j];
            for (int si = 0; si < f; si++)
                svec[si] += val * xn[si];
        }
        //gettimeofday(&t2, NULL);
        //*time_prepareb += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // solve the system of Ax=b, and get x = the uth row of X
        //gettimeofday(&t1, NULL);
        int cgiter = 0;
        //cg(smat, yi, svec, f, &cgiter, 100, 0.00001);
        cg_opt(smat, yi, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        //cholesky(smat, yi, svec, f);
        //gettimeofday(&t2, NULL);
        //printf("i = %i, cgiter = %i\n", i, cgiter);
        //*time_solver += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    free(smat_omp);
    free(svec_omp);
    //free(YT);
    free(residual_omp);
    free(y_omp);
    free(p_omp);
    free(q_omp);
}

void als_recsys_sparse(int *csrRowPtrR, int *csrColIdxR, float *csrValR,
                       int *cscColPtrR, int *cscRowIdxR, float *cscValR,
                       int nnz, float *X, float *Y, double *erroraccum,
                       int m, int n, int f, float lamda, int nthreads)
{
    // create YT and Rp
    //float *YT = (float *)malloc(sizeof(float) * f * n);
    //float *Rp = (float *)malloc(sizeof(float) * m * n);

    int iter = 0;
    double error = 0.0;
    double error_old = 0.0;
    double error_new = 0.0;
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

    do
    {
        // step 1. update X
        gettimeofday(&t1, NULL);
        updateX_recsys_sparse(csrRowPtrR, csrColIdxR, csrValR, nnz, X, Y, m, n, f, lamda, nthreads,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
//return;
        // step 2. update Y
        gettimeofday(&t1, NULL);
        updateY_recsys_sparse(cscColPtrR, cscRowIdxR, cscValR, nnz, X, Y, m, n, f, lamda, nthreads,
                       &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 3. validate
        // step 3-1. matrix multiplication
        gettimeofday(&t1, NULL);
        //matmat_transB(Rp, X, Y, m, f, n);

        memset(erroraccum, 0, sizeof(double) * nthreads);
        // step 3-2. calculate error

        //int nnz = 0;
        //for (int i = 0; i < m * n; i++)
        //{
        //    if (R[i] != 0)
        //    {
        //        error_new += fabs(Rp[i] - R[i]) * fabs(Rp[i] - R[i]);
        //        nnz++;
        //    }
        //}
        //
        #pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            //erroraccum[i] = 0.0;
            int tid = omp_get_thread_num();
            for (int j = csrRowPtrR[i]; j < csrRowPtrR[i+1]; j++)
            {
                int col = csrColIdxR[j];
                float val = csrValR[j];
                float rpij = 0.0;
                for (int k = 0; k < f; k++)
                    rpij += X[i * f + k] * Y[col * f + k];
                erroraccum[tid] += fabs(rpij - val) * fabs(rpij - val);
                //erroraccum[i] += fabs(rpij - val) * fabs(rpij - val);
                //error_new += fabs(rpij - val) * fabs(rpij - val);
                //printf("[%i][%i] r = %4.2f rp = %4.2f\n", i, col, val, rpij);
            }
        }

        error_new = 0.0;
        for (int i = 0; i < nthreads; i++)
            error_new += erroraccum[i];
        error_new = sqrt(error_new/nnz);

        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        error = fabs(error_new - error_old);// / error_new;
        error_old = error_new;
        printf("iter = %i, error_new = %f\n", iter, error_new);

        iter++;
    }
    while(iter < 4);
    //while(iter < 1000 && error_new > 0.794735); 
    //while(iter < 1000 && error > 0.00001);
    //while(iter < 1000 && error > 0.918025);

    //printf("\nR = \n");
    //printmat(R, m, n);

    //printf("\nRp = \n");
    //printmat(Rp, m, n);
/*
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrR[i]; j < csrRowPtrR[i+1]; j++)
            {
                int col = csrColIdxR[j];
                float val = csrValR[j];
                float rpij = 0.0;
                for (int k = 0; k < f; k++)
                    rpij += X[i * f + k] * Y[col * f + k];
                error_new += fabs(rpij - val) * fabs(rpij - val);
                //printf("final [%i][%i] r = %4.2f rp = %4.2f\n", i, col, val, rpij);
            }
        }
*/
    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);

    //free(Rp);
    //free(YT);
    //free(error_new_omp);
}

int binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

void updateX_recsys_sparse_balanced(int *csrRowPtrR, int *csrColIdxR, float *csrValR, int nnz, float *X, float *Y,
                    int *csrSplitter, int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    #pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        float *residual = (float *)malloc(sizeof(float) * f);
        float *y = (float *)malloc(sizeof(float) * f);
        float *p = (float *)malloc(sizeof(float) * f);
        float *q = (float *)malloc(sizeof(float) * f);

        // malloc smat (A) and svec (b)
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
        {
            float *xu = &X[u * f];

            memset(smat, 0, sizeof(float) * f * f);
            for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
            {
                float *yn = &Y[csrColIdxR[j] * f];
                for (int si = 0; si < f; si++)
                {
                    float val = yn[si];
                    for (int k = si; k < f; k++)
                    {
                        smat[si * f + k] += val * yn[k];
                    }
                }
            }
            for (int si = 0; si < f; si++)
            {
                smat[si * f + si] += lamda;
                for (int sj = si+1; sj < f; sj++)
                {
                    smat[sj * f + si] = smat[si * f + sj];
                }
            }

            // compute b (i.e., svec) by multiplying sYT and the uth row of R
            memset(svec, 0, sizeof(float) * f);
            for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
            {
                float *yn = &Y[csrColIdxR[j] * f];
                float val = csrValR[j];
                for (int si = 0; si < f; si++)
                    svec[si] += val * yn[si];
            }

            // solve the system of Ax=b, and get x = the uth row of X
            int cgiter = 0;
            cg_opt(smat, xu, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        }

        free(smat);
        free(svec);
        free(residual);
        free(y);
        free(p);
        free(q);
    }
}

void updateY_recsys_sparse_balanced(int *cscColPtrR, int *cscRowIdxR, float *cscValR, int nnz, float *X, float *Y,
                    int *cscSplitter, int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    #pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        float *residual = (float *)malloc(sizeof(float) * f);
        float *y = (float *)malloc(sizeof(float) * f);
        float *p = (float *)malloc(sizeof(float) * f);
        float *q = (float *)malloc(sizeof(float) * f);

        // malloc smat (A) and svec (b)
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        for (int i = cscSplitter[tid]; i < cscSplitter[tid+1]; i++)
        {
            float *yi = &Y[i * f];

            memset(smat, 0, sizeof(float) * f * f);
            for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
            {
                float *xn = &X[cscRowIdxR[j] * f];
                for (int si = 0; si < f; si++)
                {
                    float val = xn[si];
                    for (int k = si; k < f; k++)
                    {
                        smat[si * f + k] += val * xn[k];
                    }
                }
            }
            for (int si = 0; si < f; si++)
            {
                smat[si * f + si] += lamda;
                for (int sj = si+1; sj < f; sj++)
                {
                    smat[sj * f + si] = smat[si * f + sj];
                }
            }

            memset(svec, 0, sizeof(float) * f);
            for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
            {
                float *xn = &X[cscRowIdxR[j] * f];
                float val = cscValR[j];
                for (int si = 0; si < f; si++)
                    svec[si] += val * xn[si];
            }

            int cgiter = 0;
            cg_opt(smat, yi, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        }

        free(smat);
        free(svec);
        free(residual);
        free(y);
        free(p);
        free(q);
    }
}

void als_recsys_sparse_balanced(int *csrRowPtrR, int *csrColIdxR, float *csrValR,
                       int *cscColPtrR, int *cscRowIdxR, float *cscValR,
                       int nnz, float *X, float *Y, double *erroraccum,
                       int *csrSplitter, int *cscSplitter,
                       int m, int n, int f, float lamda, int nthreads)
{
    int iter = 0;
    double error = 0.0;
    double error_old = 0.0;
    double error_new = 0.0;
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

    do
    {
        // step 1. update X
        gettimeofday(&t1, NULL);
        updateX_recsys_sparse_balanced(csrRowPtrR, csrColIdxR, csrValR, nnz, 
                       X, Y, csrSplitter, m, n, f, lamda, nthreads,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 2. update Y
        gettimeofday(&t1, NULL);
        updateY_recsys_sparse_balanced(cscColPtrR, cscRowIdxR, cscValR, nnz, 
                       X, Y, cscSplitter, m, n, f, lamda, nthreads,
                       &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 3. validate
        // step 3-1. calculate error
        gettimeofday(&t1, NULL);

        #pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
        {
            erroraccum[tid] = 0.0;
            for (int i = csrSplitter[tid]; i < csrSplitter[tid+1]; i++)
            {
                for (int j = csrRowPtrR[i]; j < csrRowPtrR[i+1]; j++)
                {
                    int col = csrColIdxR[j];
                    float val = csrValR[j];
                    float rpij = 0.0;
                    for (int k = 0; k < f; k++)
                        rpij += X[i * f + k] * Y[col * f + k];
                    erroraccum[tid] += fabs(rpij - val) * fabs(rpij - val);
                }
            }
        }

        error_new = 0.0;
        for (int i = 0; i < nthreads; i++)
            error_new += erroraccum[i];
        error_new = sqrt(error_new/nnz);

        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        error = fabs(error_new - error_old);
        error_old = error_new;
        printf("iter = %i, error_new = %f\n", iter, error_new);

        iter++;
    }
    while(iter < 4);

    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);
}

void outerproduct(float *vec, float *mat, int f)
{
    for (int si = 0; si < f; si++)
    {
        float val = vec[si];
        for (int k = si; k < f; k++)
        {
            mat[si * f + k] += val * vec[k];
        }
    }
}

void outerproduct_avx512(float *vec, float *mat, int f)
{
    /*for (int si = 0; si < 64; si++)
    {
        float val = vec[si];
        for (int k = si; k < f; k++)
        {
            mat[si * f + k] += val * vec[k];
        }
    }*/

    __m512 vv0, vv1, vv2, vv3, scale, d1;

    vv0 = _mm512_loadu_ps(vec);
    vv1 = _mm512_loadu_ps(&vec[16]);
    vv2 = _mm512_loadu_ps(&vec[32]);
    vv3 = _mm512_loadu_ps(&vec[48]);
    
    // blocks 00, 01, 02, 03 
    for (int i = 0; i < 16; i++)
    {
        scale = _mm512_set1_ps(vec[i]);

        // block 00: vv0 vv0
        d1 = _mm512_loadu_ps(&mat[i * 64]);
        d1 = _mm512_fmadd_ps(scale, vv0, d1);
        _mm512_storeu_ps (&mat[i * 64], d1);

        // block 01: vv0 vv1
        d1 = _mm512_loadu_ps(&mat[i * 64 + 16]);
        d1 = _mm512_fmadd_ps(scale, vv1, d1);
        _mm512_storeu_ps (&mat[i * 64 + 16], d1);

        // block 02: vv0 vv2
        d1 = _mm512_loadu_ps(&mat[i * 64 + 32]);
        d1 = _mm512_fmadd_ps(scale, vv2, d1);
        _mm512_storeu_ps (&mat[i * 64 + 32], d1);

        // block 03: vv0 vv3
        d1 = _mm512_loadu_ps(&mat[i * 64 + 48]);
        d1 = _mm512_fmadd_ps(scale, vv3, d1);
        _mm512_storeu_ps (&mat[i * 64 + 48], d1);
    }
    
    // blocks 11, 12, 13 
    for (int i = 16; i < 32; i++)
    {
        scale = _mm512_set1_ps(vec[i]);

        // block 11: vv1 vv1
        d1 = _mm512_loadu_ps(&mat[i * 64 + 16]);
        d1 = _mm512_fmadd_ps(scale, vv1, d1);
        _mm512_storeu_ps (&mat[i * 64 + 16], d1);

        // block 12: vv1 vv2
        d1 = _mm512_loadu_ps(&mat[i * 64 + 32]);
        d1 = _mm512_fmadd_ps(scale, vv2, d1);
        _mm512_storeu_ps (&mat[i * 64 + 32], d1);

        // block 13: vv1 vv3
        d1 = _mm512_loadu_ps(&mat[i * 64 + 48]);
        d1 = _mm512_fmadd_ps(scale, vv3, d1);
        _mm512_storeu_ps (&mat[i * 64 + 48], d1);
    }

    // blocks 22, 23 
    for (int i = 32; i < 48; i++)
    {
        scale = _mm512_set1_ps(vec[i]);

        // block 22: vv2 vv2
        d1 = _mm512_loadu_ps(&mat[i * 64 + 32]);
        d1 = _mm512_fmadd_ps(scale, vv2, d1);
        _mm512_storeu_ps (&mat[i * 64 + 32], d1);

        // block 23: vv2 vv3
        d1 = _mm512_loadu_ps(&mat[i * 64 + 48]);
        d1 = _mm512_fmadd_ps(scale, vv3, d1);
        _mm512_storeu_ps (&mat[i * 64 + 48], d1);
    }

    // block 33 
    for (int i = 48; i < 64; i++)
    {
        scale = _mm512_set1_ps(vec[i]);

        // block 33: vv3 vv3
        d1 = _mm512_loadu_ps(&mat[i * 64 + 48]);
        d1 = _mm512_fmadd_ps(scale, vv3, d1);
        _mm512_storeu_ps (&mat[i * 64 + 48], d1);
    }
}

void updateX_recsys_sparse_balanced_avx512(int *csrRowPtrR, int *csrColIdxR, float *csrValR, int nnz, float *X, float *Y,
                    int *csrSplitter, int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    //double time = 0;
    #pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        float *residual = (float *)malloc(sizeof(float) * f);
        float *y = (float *)malloc(sizeof(float) * f);
        float *p = (float *)malloc(sizeof(float) * f);
        float *q = (float *)malloc(sizeof(float) * f);

        // malloc smat (A) and svec (b)
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
        {
            float *xu = &X[u * f];

            memset(smat, 0, sizeof(float) * f * f);
            //struct timeval t1, t2;
            //gettimeofday(&t1, NULL);
            for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
            {
                //float *yn = &Y[csrColIdxR[j] * f];
                //for (int si = 0; si < f; si++)
                //{
                //    float val = yn[si];
                //    for (int k = si; k < f; k++)
                //    {
                //        smat[si * f + k] += val * yn[k];
                //    }
                //}
                outerproduct_avx512(&Y[csrColIdxR[j] * f], smat, f);
            }
            //gettimeofday(&t2, NULL);
            //time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            for (int si = 0; si < f; si++)
            {
                smat[si * f + si] += lamda;
                for (int sj = si+1; sj < f; sj++)
                {
                    smat[sj * f + si] = smat[si * f + sj];
                }
            }

            // compute b (i.e., svec) by multiplying sYT and the uth row of R
            memset(svec, 0, sizeof(float) * f);
            for (int j = csrRowPtrR[u]; j < csrRowPtrR[u+1]; j++)
            {
                float *yn = &Y[csrColIdxR[j] * f];
                float val = csrValR[j];
                for (int si = 0; si < f; si++)
                    svec[si] += val * yn[si];
            }

            // solve the system of Ax=b, and get x = the uth row of X
            int cgiter = 0;
            cg_opt_avx512(smat, xu, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        }

        free(smat);
        free(svec);
        free(residual);
        free(y);
        free(p);
        free(q);
    }
    //printf("updateX_recsys_sparse_balanced_avx512 outerproduct time = %f\n", time);
}

void updateY_recsys_sparse_balanced_avx512(int *cscColPtrR, int *cscRowIdxR, float *cscValR, int nnz, float *X, float *Y,
                    int *cscSplitter, int m, int n, int f, float lamda, int nthreads,
                    double *time_prepareA, double *time_prepareb, double *time_solver)
{
    //double time = 0;
    #pragma omp parallel for
    for (int tid = 0; tid < nthreads; tid++)
    {
        float *residual = (float *)malloc(sizeof(float) * f);
        float *y = (float *)malloc(sizeof(float) * f);
        float *p = (float *)malloc(sizeof(float) * f);
        float *q = (float *)malloc(sizeof(float) * f);

        // malloc smat (A) and svec (b)
        float *smat = (float *)malloc(sizeof(float) * f * f);
        float *svec = (float *)malloc(sizeof(float) * f);

        for (int i = cscSplitter[tid]; i < cscSplitter[tid+1]; i++)
        {
            float *yi = &Y[i * f];

            memset(smat, 0, sizeof(float) * f * f);
            //struct timeval t1, t2;
            //gettimeofday(&t1, NULL);
            for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
            {
                //float *xn = &X[cscRowIdxR[j] * f];
                //for (int si = 0; si < f; si++)
                //{
                //    float val = xn[si];
                //    for (int k = si; k < f; k++)
                //    {
                //        smat[si * f + k] += val * xn[k];
                //    }
                //}
                outerproduct_avx512(&X[cscRowIdxR[j] * f], smat, f);
            }
            //gettimeofday(&t2, NULL);
            //time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            for (int si = 0; si < f; si++)
            {
                smat[si * f + si] += lamda;
                for (int sj = si+1; sj < f; sj++)
                {
                    smat[sj * f + si] = smat[si * f + sj];
                }
            }

            memset(svec, 0, sizeof(float) * f);
            for (int j = cscColPtrR[i]; j < cscColPtrR[i+1]; j++)
            {
                float *xn = &X[cscRowIdxR[j] * f];
                float val = cscValR[j];
                for (int si = 0; si < f; si++)
                    svec[si] += val * xn[si];
            }

            int cgiter = 0;
            cg_opt_avx512(smat, yi, svec, residual, y, p, q, f, &cgiter, 100, 0.00001);
        }

        free(smat);
        free(svec);
        free(residual);
        free(y);
        free(p);
        free(q);
    }
    //printf("updateY_recsys_sparse_balanced_avx512 outerproduct time = %f\n", time);
}

void als_recsys_sparse_balanced_avx512(int *csrRowPtrR, int *csrColIdxR, float *csrValR,
                       int *cscColPtrR, int *cscRowIdxR, float *cscValR,
                       int nnz, float *X, float *Y, double *erroraccum,
                       int *csrSplitter, int *cscSplitter,
                       int m, int n, int f, float lamda, int nthreads)
{
    int iter = 0;
    double error = 0.0;
    double error_old = 0.0;
    double error_new = 0.0;
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

    do
    {
        // step 1. update X
        gettimeofday(&t1, NULL);
        updateX_recsys_sparse_balanced_avx512(csrRowPtrR, csrColIdxR, csrValR, nnz, 
                       X, Y, csrSplitter, m, n, f, lamda, nthreads,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 2. update Y
        gettimeofday(&t1, NULL);
        updateY_recsys_sparse_balanced_avx512(cscColPtrR, cscRowIdxR, cscValR, nnz, 
                       X, Y, cscSplitter, m, n, f, lamda, nthreads,
                       &time_updatey_prepareA, &time_updatey_prepareb, &time_updatey_solver);
        gettimeofday(&t2, NULL);
        time_updatey += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 3. validate
        // step 3-1. calculate error
        gettimeofday(&t1, NULL);

        #pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
        {
            erroraccum[tid] = 0.0;
            for (int i = csrSplitter[tid]; i < csrSplitter[tid+1]; i++)
            {
                for (int j = csrRowPtrR[i]; j < csrRowPtrR[i+1]; j++)
                {
                    int col = csrColIdxR[j];
                    float val = csrValR[j];
                    //float rpij = 0.0;
                    //for (int k = 0; k < f; k++)
                    //    rpij += X[i * f + k] * Y[col * f + k];
                    float rpij = dotproduct_avx512(&X[i * f], &Y[col * f], f);
                    /*__m512 r0,c0,d1;

    d1 = _mm512_setzero_ps();

    r0 = _mm512_loadu_ps(&X[i * f]);
    c0 = _mm512_loadu_ps(&Y[col * f]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                   

    r0 = _mm512_loadu_ps(&X[i * f + 16]);
    c0 = _mm512_loadu_ps(&Y[col * f + 16]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                    

    r0 = _mm512_loadu_ps(&X[i * f + 32]);
    c0 = _mm512_loadu_ps(&Y[col * f + 32]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                    

    r0 = _mm512_loadu_ps(&X[i * f + 48]);
    c0 = _mm512_loadu_ps(&Y[col * f + 48]);
    d1 = _mm512_fmadd_ps(r0, c0, d1);
                   
    float rpij = _mm512_reduce_add_ps(d1);*/

                    erroraccum[tid] += fabs(rpij - val) * fabs(rpij - val);
                }
            }
        }

        error_new = 0.0;
        for (int i = 0; i < nthreads; i++)
            error_new += erroraccum[i];
        error_new = sqrt(error_new/nnz);

        gettimeofday(&t2, NULL);
        time_validate += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        error = fabs(error_new - error_old);
        error_old = error_new;
        printf("iter = %i, error_new = %f\n", iter, error_new);

        iter++;
    }
    while(iter < 4);

    printf("\nUpdate X %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatex, time_updatex_prepareA, time_updatex_prepareb, time_updatex_solver);
    printf("Update Y %4.2f ms (prepare A %4.2f ms, prepare b %4.2f ms, solver %4.2f ms)\n",
           time_updatey, time_updatey_prepareA, time_updatey_prepareb, time_updatey_solver);
    printf("Validate %4.2f ms\n", time_validate);
}

int main(int argc, char ** argv)
{
    // parameters
    int f = 0;

    // method
    char *method = argv[1];
    printf("\n");

    char *filename = argv[2];
    printf ("filename = %s\n", filename);

    int m, n, nnzR, isSymmetricR;

    mmio_info(&m, &n, &nnzR, &isSymmetricR, filename);
    int *csrRowPtrR = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxR = (int *)malloc(nnzR * sizeof(int));
    float *csrValR    = (float *)malloc(nnzR * sizeof(float));
    mmio_data(csrRowPtrR, csrColIdxR, csrValR, filename);

    //FILE *file;
    //file = fopen(filename, "r+");
    //fscanf(file, "%i", &m);
    //fscanf(file, "%i", &n);
    printf("The order of the rating matrix R is %i by %i, #nonzeros = %i\n",
           m, n, nnzR);

    // create R
    //float *R = (float *)malloc(sizeof(float) * m * n);
    //memset(R, 0, sizeof(float) * m * n);

    // read R
    //for (int i = 0; i < m; i++)
    //    for (int j = 0; j < n; j++)
    //        fscanf(file, "%f", &R[i * n + j]);

    //printf("\nR = \n");
    //printmat(R, m, n);

    f = atoi(argv[3]);
    printf("The latent feature is %i \n", f);

    // create X
    float *X = (float *)malloc(sizeof(float) * m * f);
    memset(X, 0, sizeof(float) * m * f);

    // create Y
    float *Y = (float *)malloc(sizeof(float) * n * f);
    for (int i = 0; i < n * f; i++)
        Y[i] = 1; //i % 2 + 1;

    // lamda parameter
    float lamda = 0.1;

    // call function
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    if (strcmp(method, "als") == 0)
    {
        //als(R, X, Y, m, n, f, lamda);
    }
    else if (strcmp(method, "als_recsys") == 0)
    {
        float *R = (float *)malloc(sizeof(float) * m * n);
        memset(R, 0, sizeof(float) * m * n);

        // put nonzeros into the dense form of R (the array R)
        for (int rowidx = 0; rowidx < m; rowidx++)
        {
            for (int j = csrRowPtrR[rowidx]; j < csrRowPtrR[rowidx + 1]; j++)
            {
                int colidx = csrColIdxR[j];
                float val = csrValR[j];
                R[rowidx * n + colidx] = val;
            }
        }

        als_recsys(R, X, Y, m, n, f, lamda);

        free(R);
    }
    else if (strcmp(method, "als_recsys_sparse") == 0)
    {
        int nthreads = atoi(argv[4]);
        omp_set_num_threads(nthreads);
        printf("#threads is %i \n", nthreads);

        int *cscColPtrR = (int *)malloc((n+1) * sizeof(int));
        int *cscRowIdxR = (int *)malloc(nnzR * sizeof(int));
        float *cscValR    = (float *)malloc(nnzR * sizeof(float));
        matrix_transposition(m, n, nnzR, csrRowPtrR, csrColIdxR, csrValR,
                         cscRowIdxR, cscColPtrR, cscValR);

        double *erroraccum = (double *)malloc(nthreads * sizeof(double));
        als_recsys_sparse(csrRowPtrR, csrColIdxR, csrValR,
                          cscColPtrR, cscRowIdxR, cscValR,
                          nnzR, X, Y, erroraccum, m, n, f, lamda, nthreads);

        free(cscColPtrR);
        free(cscRowIdxR);
        free(cscValR);
        free(erroraccum);
    }
    else if (strcmp(method, "als_recsys_sparse_balanced") == 0)
    {
        int nthreads = atoi(argv[4]);
        omp_set_num_threads(nthreads);
        printf("#threads is %i \n", nthreads);

        int *cscColPtrR = (int *)malloc((n+1) * sizeof(int));
        int *cscRowIdxR = (int *)malloc(nnzR * sizeof(int));
        float *cscValR    = (float *)malloc(nnzR * sizeof(float));
        matrix_transposition(m, n, nnzR, csrRowPtrR, csrColIdxR, csrValR,
                         cscRowIdxR, cscColPtrR, cscValR);

        // find balanced points
        int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
        int *cscSplitter = (int *)malloc((nthreads+1) * sizeof(int));

        /*int stridem = floor((double)m/(double)nthreads);
        int striden = floor((double)n/(double)nthreads);
        #pragma omp parallel for
        for (int i = 0; i < nthreads+1; i++)
        {
            csrSplitter[i] = i == nthreads ? m : (i * stridem);
            cscSplitter[i] = i == nthreads ? n : (i * striden);
        }*/

        int stridennz = ceil((double)nnzR/(double)nthreads);
        #pragma omp parallel for
        for (int tid = 0; tid <= nthreads; tid++)
        {
            // compute partition boundaries by partition of size stride
            int boundary = tid * stridennz;

            // clamp partition boundaries to [0, nnzR]
            boundary = boundary > nnzR ? nnzR : boundary;

            // binary search
            csrSplitter[tid] = binary_search_right_boundary_kernel(csrRowPtrR, boundary, m + 1) - 1;
            cscSplitter[tid] = binary_search_right_boundary_kernel(cscColPtrR, boundary, n + 1) - 1;
        }

        /*int nnzr = 0;
        for (int tid = 0; tid < nthreads; tid++)
        {
            int nzr = 0;
            for (int i = csrSplitter[tid]; i < csrSplitter[tid+1]; i++)
                nzr += csrRowPtrR[i+1] - csrRowPtrR[i];
            nnzr += nzr;
            //printf("Row dist: tid = %i, nzr = %i, nnzr = %i\n", tid, nzr, nnzr);
            printf("Row dist: %i\n", nzr);
        }

        int nnzc = 0;
        for (int tid = 0; tid < nthreads; tid++)
        {
            int nzc = 0;
            for (int i = cscSplitter[tid]; i < cscSplitter[tid+1]; i++)
                nzc += cscColPtrR[i+1] - cscColPtrR[i];
            nnzc += nzc;
            //printf("Column dist: tid = %i, nzc = %i, nnzc = %i\n", tid, nzc, nnzc);
            printf("Column dist: %i\n", nzc);
        }*/

        double *erroraccum = (double *)malloc(nthreads * sizeof(double));
        als_recsys_sparse_balanced(csrRowPtrR, csrColIdxR, csrValR,
                          cscColPtrR, cscRowIdxR, cscValR,
                          nnzR, X, Y, erroraccum, 
                          csrSplitter, cscSplitter, 
                          m, n, f, lamda, nthreads);

        free(cscColPtrR);
        free(cscRowIdxR);
        free(cscValR);
        free(erroraccum);
        free(csrSplitter);
        free(cscSplitter);
    }
    else if (strcmp(method, "als_recsys_sparse_balanced_avx512") == 0)
    {
        if (f != 64)
        {
            printf("To use AVX512 version, f has to be 64.\n Program exit.\n");
            return 0;
        }
        int nthreads = atoi(argv[4]);
        omp_set_num_threads(nthreads);
        printf("#threads is %i \n", nthreads);

        int *cscColPtrR = (int *)malloc((n+1) * sizeof(int));
        int *cscRowIdxR = (int *)malloc(nnzR * sizeof(int));
        float *cscValR    = (float *)malloc(nnzR * sizeof(float));
        matrix_transposition(m, n, nnzR, csrRowPtrR, csrColIdxR, csrValR,
                         cscRowIdxR, cscColPtrR, cscValR);

        // find balanced points
        int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
        int *cscSplitter = (int *)malloc((nthreads+1) * sizeof(int));

        int stridennz = ceil((double)nnzR/(double)nthreads);
        #pragma omp parallel for
        for (int tid = 0; tid <= nthreads; tid++)
        {
            // compute partition boundaries by partition of size stride
            int boundary = tid * stridennz;

            // clamp partition boundaries to [0, nnzR]
            boundary = boundary > nnzR ? nnzR : boundary;

            // binary search
            csrSplitter[tid] = binary_search_right_boundary_kernel(csrRowPtrR, boundary, m + 1) - 1;
            cscSplitter[tid] = binary_search_right_boundary_kernel(cscColPtrR, boundary, n + 1) - 1;
        }

        double *erroraccum = (double *)malloc(nthreads * sizeof(double));
        als_recsys_sparse_balanced_avx512(csrRowPtrR, csrColIdxR, csrValR,
                          cscColPtrR, cscRowIdxR, cscValR,
                          nnzR, X, Y, erroraccum, 
                          csrSplitter, cscSplitter, 
                          m, n, f, lamda, nthreads);

        free(cscColPtrR);
        free(cscRowIdxR);
        free(cscValR);
        free(erroraccum);
        free(csrSplitter);
        free(cscSplitter);
    }
    gettimeofday(&t2, NULL);
    float time_overall = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("time_overall = %f\n", time_overall);

    free(X);
    free(Y);
    free(csrRowPtrR);
    free(csrColIdxR);
    free(csrValR);

}

