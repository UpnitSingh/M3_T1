#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 512

void fill_matrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = rand() % 10;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int A[N][N], B[N][N], C[N][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = N / size;
    int local_A[rows][N], local_C[rows][N];

    if (rank == 0) {
        fill_matrix(A);
        fill_matrix(B);
    }

    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows * N, MPI_INT, local_A, rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    #pragma omp parallel for
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < N; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
        }

    MPI_Gather(local_C, rows * N, MPI_INT, C, rows * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Matrix multiplication complete (MPI + OpenMP).\n");

    MPI_Finalize();
    return 0;
}
