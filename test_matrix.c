/*
 * Homework 5 Tests - Pointer Operations
 * CSE 220 - Spring 2026
 *
 * Name: Aninda Saprotiv Roy
 * SBU ID: 116485388
 */

#include <criterion/criterion.h>

#include "matrix.h"

/* ------------------------- SparseMatrix tests ------------------------- */

Test(SparseMatrix, single_nonzero)
{
    int D[2] = {2, 2};
    int M[2][2] = {{0, 1}, {0, 0}};
    int S[3][2];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 1);
    cr_assert_eq(S[0][0], 0);
    cr_assert_eq(S[1][0], 1);
    cr_assert_eq(S[2][0], 1);
}

Test(SparseMatrix, exact_limit_nonzero)
{
    int D[2] = {2, 3};
    int M[2][3] = {{1, 0, 2}, {0, 3, 0}};
    int S[3][3];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 3);
    cr_assert_eq(S[2][2], 3);
}

Test(SparseMatrix, not_sparse)
{
    int D[2] = {3, 3};
    int M[3][3] = {{1, 2, 3}, {4, 0, 5}, {6, 7, 0}};
    int S[3][3];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, -1);
}

Test(SparseMatrix, all_zero)
{
    int D[2] = {3, 4};
    int M[3][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    int S[3][4];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 0);
    cr_assert_eq(S[0][3], 0);
    cr_assert_eq(S[1][3], 0);
    cr_assert_eq(S[2][3], 0);
}

Test(SparseMatrix, rectangular_sparse)
{
    int D[2] = {4, 2};
    int M[4][2] = {{4, 0}, {0, 5}, {0, 0}, {2, 0}};
    int S[3][4];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 3);
    cr_assert_eq(S[0][2], 3);
    cr_assert_eq(S[1][2], 0);
    cr_assert_eq(S[2][2], 2);
}

Test(SparseMatrix, negative_values)
{
    int D[2] = {2, 4};
    int M[2][4] = {{0, -2, 0, 0}, {3, 0, 0, -1}};
    int S[3][4];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 3);
    cr_assert_eq(S[2][0], -2);
    cr_assert_eq(S[2][2], -1);
}

Test(SparseMatrix, trailing_zero_padding)
{
    int D[2] = {1, 5};
    int M[1][5] = {{7, 0, 0, 0, 0}};
    int S[3][5];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 1);
    cr_assert_eq(S[0][4], 0);
    cr_assert_eq(S[1][4], 0);
    cr_assert_eq(S[2][4], 0);
}

Test(SparseMatrix, dense_large_dim)
{
    int D[2] = {2, 5};
    int M[2][5] = {{1, 1, 1, 1, 1}, {1, 0, 0, 0, 0}};
    int S[3][5];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, -1);
}

Test(SparseMatrix, order_preserved_row_major)
{
    int D[2] = {3, 3};
    int M[3][3] = {{0, 8, 0}, {9, 0, 0}, {0, 0, 5}};
    int S[3][3];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 3);
    cr_assert_eq(S[0][0], 0);
    cr_assert_eq(S[1][0], 1);
    cr_assert_eq(S[2][0], 8);
    cr_assert_eq(S[0][1], 1);
    cr_assert_eq(S[1][1], 0);
    cr_assert_eq(S[2][1], 9);
}

Test(SparseMatrix, single_row)
{
    int D[2] = {1, 3};
    int M[1][3] = {{0, 2, 3}};
    int S[3][3];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 2);
    cr_assert_eq(S[1][1], 2);
}

Test(SparseMatrix, single_column)
{
    int D[2] = {4, 1};
    int M[4][1] = {{0}, {4}, {0}, {1}};
    int S[3][4];

    int r = SparseMatrix(D, M, S);
    cr_assert_eq(r, 2);
    cr_assert_eq(S[0][1], 3);
    cr_assert_eq(S[2][1], 1);
}

/* ------------------------ HadamardProduct tests ----------------------- */

Test(HadamardProduct, exact_match)
{
    int D[6] = {2, 2, 2, 2, 2, 2};
    int M[2][2] = {{1, 2}, {3, 4}};
    int N[2][2] = {{2, 2}, {2, 2}};
    int A[2][2];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], 2);
    cr_assert_eq(A[1][1], 8);
}

Test(HadamardProduct, oversized_A)
{
    int D[6] = {2, 2, 2, 2, 3, 3};
    int M[2][2] = {{1, 2}, {3, 4}};
    int N[2][2] = {{5, 6}, {7, 8}};
    int A[3][3];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, 2);
    cr_assert_eq(A[0][0], 5);
    cr_assert_eq(A[1][1], 32);
    cr_assert_eq(A[2][2], 0);
}

Test(HadamardProduct, partial_with_compatible_inputs)
{
    int D[6] = {2, 3, 2, 3, 1, 2};
    int M[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int N[2][3] = {{2, 2, 2}, {3, 3, 3}};
    int A[1][2];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -3);
    cr_assert_eq(A[0][0], 2);
    cr_assert_eq(A[0][1], 4);
}

Test(HadamardProduct, incompatible_but_A_enough)
{
    int D[6] = {3, 3, 2, 2, 2, 2};
    int M[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int N[2][2] = {{2, 5}, {1, 4}};
    int A[2][2];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[0][0], 2);
    cr_assert_eq(A[0][1], 10);
    cr_assert_eq(A[1][0], 4);
    cr_assert_eq(A[1][1], 20);
}

Test(HadamardProduct, incompatible_partial)
{
    int D[6] = {3, 3, 2, 2, 1, 1};
    int M[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int N[2][2] = {{2, 5}, {1, 4}};
    int A[1][1];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -2);
    cr_assert_eq(A[0][0], 2);
}

Test(HadamardProduct, incompatible_rows)
{
    int D[6] = {1, 4, 3, 4, 1, 4};
    int M[1][4] = {{1, 2, 3, 4}};
    int N[3][4] = {{2, 2, 2, 2}, {9, 9, 9, 9}, {8, 8, 8, 8}};
    int A[1][4];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[0][3], 8);
}

Test(HadamardProduct, incompatible_cols)
{
    int D[6] = {2, 4, 2, 2, 2, 2};
    int M[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    int N[2][2] = {{2, 3}, {4, 5}};
    int A[2][2];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[1][1], 30);
}

Test(HadamardProduct, zero_initialization_check)
{
    int D[6] = {2, 2, 2, 2, 3, 2};
    int M[2][2] = {{1, 1}, {1, 1}};
    int N[2][2] = {{1, 1}, {1, 1}};
    int A[3][2] = {{9, 9}, {9, 9}, {9, 9}};

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, 2);
    cr_assert_eq(A[2][0], 0);
    cr_assert_eq(A[2][1], 0);
}

Test(HadamardProduct, negative_entries)
{
    int D[6] = {2, 2, 2, 2, 2, 2};
    int M[2][2] = {{-1, 2}, {-3, 4}};
    int N[2][2] = {{5, -6}, {7, -8}};
    int A[2][2];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], -5);
    cr_assert_eq(A[0][1], -12);
    cr_assert_eq(A[1][0], -21);
    cr_assert_eq(A[1][1], -32);
}

Test(HadamardProduct, min_overlap_only)
{
    int D[6] = {4, 1, 1, 4, 1, 1};
    int M[4][1] = {{2}, {3}, {4}, {5}};
    int N[1][4] = {{7, 8, 9, 10}};
    int A[1][1];

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[0][0], 14);
}

Test(HadamardProduct, empty_overlap_not_possible_with_vla_sizes)
{
    int D[6] = {1, 1, 1, 1, 1, 1};
    int M[1][1] = {{9}};
    int N[1][1] = {{3}};
    int A[1][1] = {{5}};

    int r = HadamardProduct(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], 27);
}

/* ------------------------- Multiplication tests ----------------------- */

Test(Multiplication, exact_compatible)
{
    int D[6] = {2, 2, 2, 2, 2, 2};
    int M[2][2] = {{1, 2}, {3, 4}};
    int N[2][2] = {{5, 6}, {7, 8}};
    int A[2][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], 19);
    cr_assert_eq(A[0][1], 22);
    cr_assert_eq(A[1][0], 43);
    cr_assert_eq(A[1][1], 50);
}

Test(Multiplication, oversized_A_compatible)
{
    int D[6] = {2, 3, 3, 2, 3, 3};
    int M[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int N[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int A[3][3];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 2);
    cr_assert_eq(A[0][0], 22);
    cr_assert_eq(A[1][1], 64);
    cr_assert_eq(A[2][2], 0);
}

Test(Multiplication, partial_A_compatible)
{
    int D[6] = {2, 3, 3, 2, 1, 2};
    int M[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int N[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int A[1][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, -3);
    cr_assert_eq(A[0][0], 22);
    cr_assert_eq(A[0][1], 28);
}

Test(Multiplication, incompatible_but_full_A)
{
    int D[6] = {3, 2, 3, 2, 3, 2};
    int M[3][2] = {{1, 2}, {4, 3}, {7, 8}};
    int N[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int A[3][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[0][0], 7);
    cr_assert_eq(A[0][1], 10);
    cr_assert_eq(A[1][0], 13);
    cr_assert_eq(A[1][1], 20);
    cr_assert_eq(A[2][0], 31);
    cr_assert_eq(A[2][1], 46);
}

Test(Multiplication, incompatible_partial_A)
{
    int D[6] = {3, 2, 3, 2, 2, 1};
    int M[3][2] = {{1, 2}, {4, 3}, {7, 8}};
    int N[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int A[2][1];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, -2);
    cr_assert_eq(A[0][0], 7);
    cr_assert_eq(A[1][0], 13);
}

Test(Multiplication, incompatible_3x3_with_2x2)
{
    int D[6] = {3, 3, 2, 2, 3, 2};
    int M[3][3] = {{1, 2, 3}, {4, 3, 6}, {7, 8, 9}};
    int N[2][2] = {{1, 2}, {3, 4}};
    int A[3][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, -1);
    cr_assert_eq(A[0][0], 7);
    cr_assert_eq(A[0][1], 10);
    cr_assert_eq(A[1][0], 13);
    cr_assert_eq(A[1][1], 20);
    cr_assert_eq(A[2][0], 31);
    cr_assert_eq(A[2][1], 46);
}

Test(Multiplication, zero_matrix)
{
    int D[6] = {2, 3, 3, 2, 2, 2};
    int M[2][3] = {{0, 0, 0}, {0, 0, 0}};
    int N[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int A[2][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], 0);
    cr_assert_eq(A[1][1], 0);
}

Test(Multiplication, negative_values)
{
    int D[6] = {2, 2, 2, 2, 2, 2};
    int M[2][2] = {{-1, 2}, {3, -4}};
    int N[2][2] = {{5, -6}, {-7, 8}};
    int A[2][2];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], -19);
    cr_assert_eq(A[0][1], 22);
    cr_assert_eq(A[1][0], 43);
    cr_assert_eq(A[1][1], -50);
}

Test(Multiplication, single_element)
{
    int D[6] = {1, 1, 1, 1, 1, 1};
    int M[1][1] = {{7}};
    int N[1][1] = {{9}};
    int A[1][1];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 1);
    cr_assert_eq(A[0][0], 63);
}

Test(Multiplication, a_zeroed_before_fill)
{
    int D[6] = {2, 2, 2, 2, 3, 3};
    int M[2][2] = {{1, 0}, {0, 1}};
    int N[2][2] = {{2, 3}, {4, 5}};
    int A[3][3] = {{7, 7, 7}, {7, 7, 7}, {7, 7, 7}};

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, 2);
    cr_assert_eq(A[0][0], 2);
    cr_assert_eq(A[1][1], 5);
    cr_assert_eq(A[2][2], 0);
}

Test(Multiplication, partial_overlap_shape)
{
    int D[6] = {3, 2, 2, 4, 2, 3};
    int M[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int N[2][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}};
    int A[2][3];

    int r = Multiplication(D, M, N, A);
    cr_assert_eq(r, -3);
    cr_assert_eq(A[0][0], 1);
    cr_assert_eq(A[0][1], 2);
    cr_assert_eq(A[0][2], 1);
    cr_assert_eq(A[1][0], 3);
    cr_assert_eq(A[1][1], 4);
    cr_assert_eq(A[1][2], 3);
}

/* --------------------------- DiagonalSum tests ------------------------ */

Test(DiagonalSum, exact_square)
{
    int D[4] = {3, 3, 5, 3};
    int A[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int DS[5][3];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 1);
    cr_assert_eq(DS[0][0], 15);
    cr_assert_eq(DS[0][1], 15);
    cr_assert_eq(DS[1][0], 12);
    cr_assert_eq(DS[1][1], 15);
    cr_assert_eq(DS[1][2], 18);
    cr_assert_eq(DS[2][0], 6);
    cr_assert_eq(DS[3][0], 15);
    cr_assert_eq(DS[4][0], 24);
}

Test(DiagonalSum, oversized_square)
{
    int D[4] = {3, 3, 6, 4};
    int A[3][3] = {{1, 0, 1}, {0, 2, 0}, {3, 0, 3}};
    int DS[6][4];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 2);
    cr_assert_eq(DS[0][0], 6);
    cr_assert_eq(DS[0][1], 6);
    cr_assert_eq(DS[5][3], 0);
}

Test(DiagonalSum, too_small)
{
    int D[4] = {5, 5, 2, 2};
    int A[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    };
    int DS[2][2];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, -1);
    cr_assert_eq(DS[0][0], 5);
    cr_assert_eq(DS[0][1], 5);
    cr_assert_eq(DS[1][0], 5);
    cr_assert_eq(DS[1][1], 5);
}

Test(DiagonalSum, exact_rectangular)
{
    int D[4] = {3, 2, 5, 2};
    int A[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    int DS[5][2];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 1);
    cr_assert_eq(DS[0][0], 5);
    cr_assert_eq(DS[0][1], 0);
    cr_assert_eq(DS[1][0], 9);
    cr_assert_eq(DS[1][1], 12);
    cr_assert_eq(DS[2][0], 3);
    cr_assert_eq(DS[3][0], 7);
    cr_assert_eq(DS[4][0], 11);
}

Test(DiagonalSum, partial_rectangular)
{
    int D[4] = {4, 3, 2, 2};
    int A[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    int DS[2][2];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, -1);
    cr_assert_eq(DS[0][0], 15);
    cr_assert_eq(DS[0][1], 0);
    cr_assert_eq(DS[1][0], 22);
    cr_assert_eq(DS[1][1], 26);
}

Test(DiagonalSum, one_by_one)
{
    int D[4] = {1, 1, 3, 2};
    int A[1][1] = {{9}};
    int DS[3][2];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 1);
    cr_assert_eq(DS[0][0], 9);
    cr_assert_eq(DS[0][1], 9);
    cr_assert_eq(DS[1][0], 9);
    cr_assert_eq(DS[2][0], 9);
}

Test(DiagonalSum, square_negative_values)
{
    int D[4] = {2, 2, 4, 2};
    int A[2][2] = {{-1, -2}, {-3, -4}};
    int DS[4][2];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 1);
    cr_assert_eq(DS[0][0], -5);
    cr_assert_eq(DS[0][1], -5);
    cr_assert_eq(DS[1][0], -4);
    cr_assert_eq(DS[1][1], -6);
}

Test(DiagonalSum, oversized_rectangular)
{
    int D[4] = {2, 4, 5, 6};
    int A[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    int DS[5][6];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 2);
    cr_assert_eq(DS[0][0], 7);
    cr_assert_eq(DS[1][0], 6);
    cr_assert_eq(DS[1][1], 8);
    cr_assert_eq(DS[1][2], 10);
    cr_assert_eq(DS[1][3], 12);
}

Test(DiagonalSum, no_room_for_rows)
{
    int D[4] = {3, 3, 2, 4};
    int A[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int DS[2][4];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, -1);
    cr_assert_eq(DS[0][0], 15);
    cr_assert_eq(DS[0][1], 15);
    cr_assert_eq(DS[1][2], 18);
}

Test(DiagonalSum, no_room_for_second_diag)
{
    int D[4] = {2, 2, 4, 1};
    int A[2][2] = {{3, 4}, {5, 6}};
    int DS[4][1];

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, -1);
    cr_assert_eq(DS[0][0], 9);
    cr_assert_eq(DS[1][0], 8);
    cr_assert_eq(DS[2][0], 7);
    cr_assert_eq(DS[3][0], 11);
}

Test(DiagonalSum, zero_init_check)
{
    int D[4] = {2, 3, 4, 3};
    int A[2][3] = {{1, 0, 2}, {0, 3, 0}};
    int DS[4][3] = {{9, 9, 9}, {9, 9, 9}, {9, 9, 9}, {9, 9, 9}};

    int r = DiagonalSum(D, A, DS);
    cr_assert_eq(r, 1);
    cr_assert_eq(DS[0][0], 4);
    cr_assert_eq(DS[0][1], 0);
    cr_assert_eq(DS[1][0], 1);
    cr_assert_eq(DS[1][1], 3);
    cr_assert_eq(DS[1][2], 2);
    cr_assert_eq(DS[2][0], 3);
    cr_assert_eq(DS[3][0], 3);
}