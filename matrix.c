/*
 * Homework 5 - Pointer Operations
 * CSE 220 - Spring 2026
 *
 * Name:Aninda Saprotiv Roy
 * SBU ID:116485388
 */

#include "matrix.h"

static int max(int a, int b)
{
    int result;
    result = a;
    if (b > result)
        result = b;
    return result;
}

static int min(int a, int b)
{
    int result = a;
    if (b < result)
        result = b;
    return result;
}

int SparseMatrix(int D[2], int M[D[0]][D[1]],
                 int S[3][(D[1] > D[0] ? D[1] : D[0])])
{
    int rows = D[0];
    int cols = D[1];
    int m = max(rows, cols);
    int nonZeroCount = 0;

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < m; c++) {
            *(*(S + r) + c) = 0;
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (*(*(M + i) + j) != 0) {
                nonZeroCount++;
            }
        }
    }

    if (nonZeroCount > m) {
        return -1;
    }

    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int value = *(*(M + i) + j);
            if (value != 0) {
                *(*(S + 0) + idx) = i;
                *(*(S + 1) + idx) = j;
                *(*(S + 2) + idx) = value;
                idx++;
            }
        }
    }

    return nonZeroCount;
}

int HadamardProduct(const int D[6],
                    const int M[D[0]][D[1]],
                    const int N[D[2]][D[3]],
                    int A[D[4]][D[5]])
{
    int mRows = D[0];
    int mCols = D[1];
    int nRows = D[2];
    int nCols = D[3];
    int aRows = D[4];
    int aCols = D[5];

    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < aCols; j++) {
            *(*(A + i) + j) = 0;
        }
    }

    int rLimit = min(mRows, nRows);
    int cLimit = min(mCols, nCols);
    int writeRows = min(rLimit, aRows);
    int writeCols = min(cLimit, aCols);

    for (int i = 0; i < writeRows; i++) {
        for (int j = 0; j < writeCols; j++) {
            *(*(A + i) + j) = (*(*(M + i) + j)) * (*(*(N + i) + j));
        }
    }

    if (mRows == nRows && mCols == nCols) {
        if (aRows == mRows && aCols == mCols) {
            return 1;
        }
        if (aRows >= mRows && aCols >= mCols) {
            return 2;
        }
        return -3;
    }

    if (aRows >= rLimit && aCols >= cLimit) {
        return -1;
    }

    return -2;
}

int Multiplication(const int D[6],
                   const int M[D[0]][D[1]],
                   const int N[D[2]][D[3]],
                   int A[D[4]][D[5]])
{
    /* Read all matrix dimensions from D */
    int mRows = *(D + 0), mCols = *(D + 1);
    int nRows = *(D + 2), nCols = *(D + 3);
    int aRows = *(D + 4), aCols = *(D + 5);

    /* Clear the result matrix A with zeros */
    for (int i = 0; i < aRows; i++)
        for (int j = 0; j < aCols; j++)
            *(*(A + i) + j) = 0;

    /* Determine how many rows/cols we can safely write */
    int kLimit    = min(mCols, nRows);   /* shared dimension for dot product */
    int writeRows = min(mRows, aRows);   /* don't write beyond A's row limit  */
    int writeCols = min(nCols, aCols);   /* don't write beyond A's col limit  */

    /* --- Perform Matrix Multiplication --- */
    for (int i = 0; i < writeRows; i++) {
        for (int j = 0; j < writeCols; j++) {
            int sum = 0;
            for (int k = 0; k < kLimit; k++)
                sum += (*(*(M + i) + k)) * (*(*(N + k) + j));   /* dot product of row i and col j */
            *(*(A + i) + j) = sum;
        }
    }

    /* --- Return status based on dimension compatibility --- */
    if (mCols == nRows) {
        /* Valid multiplication: M and N are compatible */
        if (aRows == mRows && aCols == nCols) return  1;  /* A is perfect size        */
        if (aRows >= mRows && aCols >= nCols) return  2;  /* A is larger than needed  */
        return -3;                                         /* A is too small           */
    }

    /* Invalid multiplication: M and N are NOT compatible */
    if (aRows >= mRows && aCols >= nCols)    return -1;   /* A fits but dims mismatch */

    return -2;                                            
}

int DiagonalSum(const int D[4],
                const int A[D[0]][D[1]],
                int DS[D[2]][D[3]])
{
    /*Read matrix dimension and output size */
    int rows = D[0];
    int cols = D[1];
    int dsRows = D[2];
    int dsCols = D[3];

    for (int i = 0; i < dsRows; i++) {
        for (int j = 0; j < dsCols; j++) {
            *(*(DS + i) + j) = 0;
        }
    }

    int mainDiag = 0;
    int antiDiag = 0;
    int diagLen = min(rows, cols);

    for (int i = 0; i < diagLen; i++) {
        mainDiag += *(*(A + i) + i);
    }

    if (rows == cols) {
        for (int i = 0; i < rows; i++) {
            antiDiag += *(*(A + i) + (cols - 1 - i));
        }
    }

    if (dsRows > 0 && dsCols > 0) {
        *(*(DS + 0) + 0) = mainDiag;
    }
    if (dsRows > 0 && dsCols > 1) {
        *(*(DS + 0) + 1) = antiDiag;
    }

    if (dsRows > 1) {
        for (int j = 0; j < min(cols, dsCols); j++) {
            int colSum = 0;
            for (int i = 0; i < rows; i++) {
                colSum += *(*(A + i) + j);
            }
            *(*(DS + 1) + j) = colSum;
        }
    }
    /*Store each rows sum in DS*/
    for (int i = 0; i < rows; i++) {
        int rowSum = 0;
        for (int j = 0; j < cols; j++) {
            rowSum += *(*(A + i) + j);
        }
        if ((i + 2) < dsRows && dsCols > 0) {
            *(*(DS + (i + 2)) + 0) = rowSum;
        }
    }

    int reqRows = rows + 2;
    int reqCols = max(cols, 2);

    if (dsRows == reqRows && dsCols == reqCols) {
        return 1;
    }
    if (dsRows >= reqRows && dsCols >= reqCols) {
        return 2;
    }

    return -1;
}