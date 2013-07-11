#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <time.h>

#define SEND_DATA_TAG 0
#define SEND_ANSWER_TAG 1

#define TILE_SIZE 24 //24
#define TILE_SIZE2 140 //140

#define min(a,b) (((a) > (b))?(b):(a))

void matmul(const float *A, const float *B, float *C, const int m, const int n, const int k)
{
	int i, j, ii, jj, ii_limit, jj_limit, l, jj_limit_minus_3;
	__m128 a_line, b_line, r_line;
	float *row_in_B, *row_in_C, ii_l_in_A;
	
	for(i = 0; i< m ; i+= TILE_SIZE) // i: row block index in C 
	{
		ii_limit = min(m, i + TILE_SIZE);

		for(j = 0 ; j < n ; j += TILE_SIZE2) // j : col block index in C
		{
			jj_limit = min(n, j + TILE_SIZE2);
			jj_limit_minus_3 = jj_limit - 3;

			for(l = 0; l < k; ++l)
			{
				row_in_B = B + l*n;
				for(ii = i ; ii < ii_limit; ++ii)
				{
					ii_l_in_A = A[ii * k + l];
					a_line = _mm_set1_ps(ii_l_in_A); //A[ii * k + l]);  
					row_in_C = C + ii * n;
					for(jj = j; jj < jj_limit_minus_3 ; jj += 4)
					{
						b_line = _mm_loadu_ps(row_in_B + jj);
						r_line = _mm_loadu_ps(row_in_C + jj);
						_mm_storeu_ps(row_in_C + jj, _mm_add_ps(_mm_mul_ps(a_line, b_line), r_line));
					}
					for(; jj < jj_limit; ++jj)
					{
						*(row_in_C + jj) += ii_l_in_A * *(row_in_B + jj);
					}
				}
			}
		}
	}
}

void matmul_mpi(float* A, float* B, float* C, int n){
	int rank, nodes_n,
		used_nodes_n,
		row_per_process, 
		i,
		j,
		k,
		start_row,
		end_row,
		child_start_row,
		child_end_row,
		*counts,
		*displs;
	
	float* my_A;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nodes_n);
	row_per_process = n / nodes_n;
	if(n % nodes_n)row_per_process ++;

	used_nodes_n = n / row_per_process;
	if(n % row_per_process)used_nodes_n ++;
		
	start_row = rank % used_nodes_n * row_per_process;
	end_row = start_row + row_per_process;
	if(end_row >= n)end_row = n;	


	if(rank == 0) {
		// 0 ~ row_per_process are mine
		counts = (int *)malloc(sizeof(int) * nodes_n);
		displs = (int *)malloc(sizeof(int) * nodes_n);
		for(i = 0 ; i <nodes_n;i++){
			child_start_row = i % used_nodes_n * row_per_process;
			child_end_row = child_start_row + row_per_process;
			if(child_end_row >= n) child_end_row = n;
			
			displs[i] = child_start_row * n;
			counts[i] = (child_end_row - child_start_row) * n;
		}
	}
	my_A = (float *)malloc(sizeof(float) * row_per_process * n);
	if(rank > 0) B = (float *)malloc(sizeof(float) * n * n);
	MPI_Scatterv(A, counts, displs, MPI_FLOAT, my_A, row_per_process * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	//자기것 계산하기
	
	if(C == NULL) {
		C = (float *)malloc(sizeof(float) * (end_row - start_row) * n);
		memset(C, 0, sizeof(float) * (end_row - start_row) * n);
	} else {
		memset(C, 0, sizeof(float) * n * n);
	}
	
	if(end_row-start_row > 0)
		matmul(my_A, B, C, end_row-start_row, n, n);
	
	// 계산 완료

	MPI_Gatherv(C, (end_row - start_row) * n, MPI_FLOAT, C, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	free(my_A);
	if(rank == 0){
		free(counts);
		free(displs);
	} else {
		free(B);
		free(C);
	}
}
