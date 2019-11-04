#ifndef recsyscsr_h
#define recsyscsr_h

void updateX_recsys_csr(int *csrRowPtrR,int *csrColIdxR,float *csrValR, float *X, float *Y,
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
        nzr = csrRowPtrR[u+1] - csrRowPtrR[u];
        // for (int k = 0; k < n; k++)
        //    nzr = R[u * n + k] == 0 ? nzr : nzr + 1;

        // malloc ru (i.e., uth row of R) and insert entries into it
        float *ru = (float *)malloc(sizeof(float) * nzr);
        
        int count = 0;
        for(int k = csrRowPtrR[u]; k < csrRowPtrR[u]+nzr; k++)
        // for(int k = csrRowPtrR[u]; k < csrRowPtrR[u+1]; k++)
        {
            ru[count] = csrValR[k];
            count++;
        }

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
        count = 0;
        for (int k = csrRowPtrR[u]; k < csrRowPtrR[u+1]; k++)
        {
            memcpy(&sY[count * f], &Y[csrColIdxR[k] * f], sizeof(float) * f);
            count++;
        }
        
        
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

void updateY_recsys_csr(int *csrRowPtrR,int *csrColIdxR,float *csrValR, float *X, float *Y,
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

    do
    {
        // step 1. update X
        gettimeofday(&t1, NULL);
        
        updateX_recsys_csr(csrRowPtrR,csrColIdxR,csrValR, X, Y, m, n, f, lamda,
                       &time_updatex_prepareA, &time_updatex_prepareb, &time_updatex_solver);
        gettimeofday(&t2, NULL);
        time_updatex += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        // step 2. update Y
        gettimeofday(&t1, NULL);
        updateY_recsys_csr(csrRowPtrR,csrColIdxR,csrValR, X, Y, m, n, f, lamda,
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
            error_new += fabs(csrValR[i] - resultXY) * fabs(csrValR[i] - resultXY);
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

    // free(Rp);
    free(YT);
}

#endif