#ifndef multiplication_h
#define multiplication_h
#define INCREMENT 100000

#include "tranpose.h"

// m:A's rows n:B's rows
void matmatCSR(float *csrValA, int *csrColIdxA, int *csrRowptrA,
               float *csrValB, int *csrColIdxB, int *csrRowptrB,
               float *csrValC, int *csrColIdxC, int *csrRowptrC,
               int m, int n, int nnzA, int nnzB)
{
    float *cscValB = (float *)malloc(sizeof(float) * nnzB);
    int *cscRowIdxB = (int *)malloc(sizeof(int) * nnzB);
    int *cscColPtrB = (int *)malloc(sizeof(int) * (m+1));

    matrix_transposition(n,m,nnzB,csrRowptrB,csrColIdxB,csrValA,cscRowIdxB,cscColPtrB,cscValB);

    int SizeValandIdx = 0;
    csrValC = (float *)malloc(sizeof(float) * (SizeValandIdx + INCREMENT));
    csrColIdxC = (int *)malloc(sizeof(int) * (SizeValandIdx + INCREMENT));
    csrRowptrC = (int *)malloc(sizeof(int) * (m+1));
    SizeValandIdx += INCREMENT;
    int nnzC = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            int idxA = 0, idxB = 0;
            // number of elem in A & B
            int nuA = csrRowptrA[i+1] - csrRowptrA[i];
            int nuB = cscColPtrB[j+1] - cscColPtrB[j];
            int valC = 0;
            while (idxA < nuA || idxB < nuB)
            {
                int col = csrColIdxA[idxA + csrRowptrA[i]];
                int row = cscRowIdxB[idxB + cscColPtrB[j]];
                if (col == row)
                {
                    valC += csrValA[idxA + csrRowptrA[i]] * cscValB[idxB + cscColPtrB[j]];
                    idxA++;
                    idxB++;
                }else if (col > row)
                {
                    idxB++;
                }else // row > col
                {
                    idxA++;
                }
            }
            if (valC == 0)
            {
                /* do nothing */
            }else
            {
                if (nnzC >= SizeValandIdx)
                {
                    csrValC = (float *)realloc(sizeof(float) * (SizeValandIdx + INCREMENT));
                    csrColIdxC = (int *)realloc(sizeof(int) * (SizeValandIdx + INCREMENT));
                    SizeValandIdx += INCREMENT;
                }
                
                csrValC[nnzC] = valC;
                csrColIdxC[nnzC] = j;
                csrRowptrC[i+1]++; // the first is 0, the add them together to get the real rowptr
                nnzC++;
            }
        }
    }
    for (int i = 1; i <= m; i++)
    {
        csrRowptrC[i] += csrRowptrC[i-1]; // add them together
    }
    
    free(cscValB);
    free(cscRowIdxB);
    free(cscColPtrB);
}

#endif /* multiplication_h */