#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

void matvec(int *csrRowPtrR,int *csrColIdxR,float *csrValR,int m,float * x,float *R);

void printvec(float *x, int n)
{
    for (int i = 0; i < n; i++)
        printf("%4.2f\n", x[i]);
}

int main()
{
    int m=4,n=4;
    int nnzR=9;
    // int *csrRowPtrR = (int *)malloc((m+1) * sizeof(int));
    // int *csrColIdxR = (int *)malloc(nnzR * sizeof(int));
    // float *csrValR    = (float *)malloc(nnzR * sizeof(float));

    int csrRowPtrR[5] = {0,2,4,7,9};
    int csrColIdxR[9] = {0,1,1,2,0,2,3,1,3};
    float csrValR[9]  = {1,7,2,8,5,3,9,6,4};

    //float *x = (float *)malloc(sizeof(float) * m );
    //memset(x, 1.0, sizeof(float) * m );
    //printvec(x, m);
    float x[4]={1,2,1,3};

    float *R = (float *)malloc(sizeof(float) * m );
    memset(R, 0, sizeof(float) * m );

    matvec(csrRowPtrR,csrColIdxR,csrValR,m,x,R);

    printvec(R, m);
}

void matvec(int *csrRowPtrR,int *csrColIdxR,float *csrValR,int m,float * x,float *R){
    // put nonzeros into the dense form of R (the array R)
    for (int rowidx = 0; rowidx < m; rowidx++)
    {
        float sum=0;
        for (int j = csrRowPtrR[rowidx]; j < csrRowPtrR[rowidx + 1]; j++)
        {
            int colidx = csrColIdxR[j];
            float val = csrValR[j];
            sum+=val*x[colidx];
            printf("val=%f ",val);
        }
        R[rowidx] = sum;
        printf("sum=%f\n",sum);
    }
}


void matmat(int *csrRowPtrR,int *csrColIdxR,float *csrValR,int m,int k,int n,int *csrRowPtrR2,int *csrColIdxR2,float *csrValR2,int *csrRowPtrR3,int *csrColIdxR3,float *csrValR3){
    // put nonzeros into the dense form of R (the array R)
    for (int rowidx = 0; rowidx < m; rowidx++)
    {
        float sum=0;
        for (int j = csrRowPtrR[rowidx]; j < csrRowPtrR[rowidx + 1]; j++)
        {
            int colidx = csrColIdxR[j];
            float val = csrValR[j];
        //     sum+=val*x[colidx];
        //     printf("val=%f ",val);
        // }
        // R[rowidx] = sum;
        // printf("sum=%f\n",sum);
    }
}