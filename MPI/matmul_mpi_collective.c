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
	fprintf(stderr, "%d: Scatter A시작\n", rank);
	MPI_Scatterv(A, counts, displs, MPI_FLOAT, my_A, row_per_process * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	fprintf(stderr, "%d: Scatter A완료\n", rank);
	fprintf(stderr, "%d: Bcast B시작\n", rank);
	MPI_Bcast(B, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	fprintf(stderr, "%d: Bcast B완료\n", rank);
	
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

	if(rank == 0)fprintf(stderr, "XXXXXXXXXXXXXXXX!\n");
	fprintf(stderr, "%d: Gather 시작\n", rank);
	MPI_Gatherv(C, (end_row - start_row) * n, MPI_FLOAT, C, counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	fprintf(stderr, "%d: Gather 완료\n", rank);
	free(my_A);
	if(rank == 0){
		free(counts);
		free(displs);
	} else {
		free(B);
		free(C);
	}
}

void init_matrix(float *A, int n){
	int i;
	for(i = 0 ; i < n * n ; i++)
		A[i] = (float)rand() / rand();
}

void check_result(float *A, float *B, float *C, int n){
	float * answer = (float *)malloc(sizeof(float) * n * n);
	int i, j, k;
	memset(answer, 0, sizeof(float) * n * n);
	for(i=0;i<n;++i){
		for(k=0;k<n;++k){
			for(j=0;j<n;++j){
				answer[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}

	for(i=0;i<n*n;++i){
		if(abs(answer[i] - C[i]) > 0.001){
			fprintf(stderr, "ERROR!!!!!\n");
			break;
		}
	}
	if(i == n * n)
		fprintf(stderr, "SUCCESS!!\n");
}


int main(int argc, char * argv[])
{
	int rank;
	int n;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	float *A, *B, *C;

	srand(time(NULL));
	n = rand() % 100 + 1; //1000; //2000;
	fprintf(stderr, "n = %d\n", n);
	if(rank == 0){
		A = (float *)malloc(sizeof(float) * n * n);
		B = (float *)malloc(sizeof(float) * n * n);
		C = (float *)malloc(sizeof(float) * n * n);
		init_matrix(A, n);
		init_matrix(B, n);
	} else{
		A = B = C = NULL;
	}
	matmul_mpi(A, B, C, n);
	if(rank == 0){
		long t = clock();
//		check_result(A, B, C, n);
		fprintf(stderr, "Check Time: %lf\n", ((double)clock() - t) / CLOCKS_PER_SEC);
	}
	MPI_Finalize();
	return 0;
}
