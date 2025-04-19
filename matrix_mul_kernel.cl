__kernel void matrix_mul(__global int* A, __global int* B, __global int* C, int N, int rows) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[i * N + k] * B[k * N + j];
    C[i * N + j] = sum;
}
