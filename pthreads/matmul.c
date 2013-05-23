#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <x86intrin.h>
#include <string.h>

#define TILE_SIZE 24 //24
#define TILE_SIZE2 140 //140

#define min(a,b) (((a) > (b))?(b):(a))

int thread_n, matrix_m, matrix_n, matrix_k, row_per_thread;
float *matrix_a, *matrix_b, *matrix_c;

void *calculate_row(void *thread_id)
{
	long id = (long)thread_id;
	int i, j, ii, jj, ii_limit, jj_limit, l, start_i, end_i, jj_limit_minus_3;
	__m128 a_line, b_line, r_line;
	const float *A = matrix_a, *B= matrix_b;
	const int n = matrix_n, m = matrix_m, k = matrix_k;
	float *row_in_B, *row_in_C, ii_l_in_A;

	start_i = row_per_thread * id;
	end_i = row_per_thread * (id + 1);
	if(end_i >= m)
		end_i = m;
	
	for(i = start_i ; i < end_i ; i += TILE_SIZE) // i: row block index in C 
	{
		ii_limit = min(end_i, i + TILE_SIZE);

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
					row_in_C = matrix_c + ii * n;
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
	pthread_exit(NULL);
}


void matmul(float *A, float *B, float *C, int m, int n, int k, int core_n)
{
	pthread_t threads[32];
	pthread_attr_t attr;
	cpu_set_t cpuset;
	void * status;
	int i, rc;

	thread_n = core_n;
	matrix_a = A;
	matrix_b = B;
	matrix_c = C;
	matrix_n = n;
	matrix_m = m;
	matrix_k = k;

	
	memset(C, 0, m*n*sizeof(float));

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	row_per_thread = m / thread_n;
	if(m % thread_n > 0)row_per_thread += 1;

	for(i = 0 ; i < thread_n; ++i)
	{
		rc = pthread_create(&threads[i], &attr, calculate_row, (void *)i);
		if(rc){
			fprintf(stderr, "ERROR: cannot create threads!");
			exit(-1);
		}
		CPU_ZERO(&cpuset);
		CPU_SET(thread_n - 1 - i, &cpuset);
		rc = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
		if(rc!=0){
		}
	}
	pthread_attr_destroy(&attr);
	for(i = 0 ; i <thread_n; ++i)
	{
		rc = pthread_join(threads[i], &status);
		if (rc) {
			fprintf(stderr, "ERROR: return code from pthread_join() was %d\n", rc);
			exit(-1);
		}
	}
}

///////////////////////////////////////////////////
/// test for correctness & performance
//////////////////////////////////////////////////

#include <stdlib.h>
#include <time.h>


int random_interval(int a, int b)
{
	return rand() % (b-a) + a;
}

void generate(float *array, int number)
{
	int i;
	for(i=0;i<number;i++){
		array[i] = (float)rand() / rand();
	}
}

void print(float *array, int n, int m){
	int i, j;
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			printf("%0.3f ",array[i * m + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void naive_matmul(float *A, float *B, float *C, int m, int n, int k)
{
	int i, j, l;
	
	for(i=0;i<m*n;++i)C[i] = 0;
	for(l = 0; l < k; ++l)
	{
		for(i = 0 ; i < m; ++i)
		{
			for(j = 0 ; j < n; ++j)
			{
				C[i * n + j] += A[i * k + l] * B[l * n +j];
			}
		}
	}
}

int main()
{
	int m, n, k;
	float *A, *B, *C, *D;
	long t;
	t = clock();


	srand(time(NULL));
	m = random_interval(1, 1500);
	n = random_interval(1, 1500);
	k = random_interval(1, 1500);
	
	fprintf(stderr, "m = %d, n = %d, k = %d\n", m, n, k);
	
	A = (float *)malloc(sizeof(float) * m * k);
	B = (float *)malloc(sizeof(float) * k * n);
	C = (float *)malloc(sizeof(float) * m * n);
	D = (float *)malloc(sizeof(float) * m * n);

	generate(A, m * k);
	generate(B, k * n);
	fprintf(stderr, "Generate Matrix : %gs\n", (double)(clock() - t) / CLOCKS_PER_SEC);
	
	t = clock();
	matmul(A, B, C, m, n, k, 32);
	fprintf(stderr, "Multiply : %gs\n", (double)(clock() - t) / CLOCKS_PER_SEC);
	
	t = clock();
	naive_matmul(A, B, D, m, n, k);
	fprintf(stderr, "Get Correct Answer : %gs\n", (double)(clock() - t) / CLOCKS_PER_SEC);

	t = clock();

	float diff;
	int flag = 0, i;
	for(i=0;i<m*n;++i)
	{	
		diff = C[i] - D[i];
		if(diff<0)diff=-diff;
		if(diff > 0.0001)
		{
			flag = 1;
			fprintf(stderr, "WRONG!!!!! %f != %f\n", C[i], D[i]);
			break;
		}
	}
	if(!flag)
		fprintf(stderr, "CORRECT!!!!!\n");

	free(A);
	free(B);
	free(C);
	free(D);
	fprintf(stderr, "Clean Up : %gs\n", (double)(clock() - t) / CLOCKS_PER_SEC);
	return 0;
}
