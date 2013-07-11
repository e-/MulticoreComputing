#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void histogram(int * data, int n, int * bins, int k){
	int num_bins = 800 / k;
	int number_of_threads = omp_get_max_threads();
	int local_bins[number_of_threads + 1][1024]; //for false sharing 
	memset(bins, 0, sizeof(bins));
	memset(local_bins, 0, sizeof(local_bins));
#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int i, j;
		
		#pragma omp for
		for(i = 0 ; i < n ; ++i)
			local_bins[id][data[i] / k] ++;

		#pragma omp for
		for(i = 0 ; i < num_bins; ++i)
		{
			for(j = 0 ; j < number_of_threads; ++j)
			{
				bins[i] += local_bins[j][i];
			}
		}
	}
}

void naive_histogram(int * data, int n, int * bins, int k){
	int num_bins = 800 / k;
	int i;
	memset(bins, 0, sizeof(bins));
	for(i=0;i<n;i++)bins[data[i] / k] ++;
}

int main()
{
	int n, k, *array, *bins, i, *bins2;
	long c;

	n = 4 * 10000 * 10000; // * 10000;
	k = 10;
	srand(time(NULL));
	c = clock();
	array = (int *)malloc(sizeof(int) * n);
	bins = (int *)malloc(sizeof(int) * 800 / k);
	bins2 = (int *)malloc(sizeof(int) * 800 / k);
	for(i=0;i<n;++i)
		array[i] = rand() % 800;
	printf("%lf\n", ((double)clock() - c) / CLOCKS_PER_SEC);
	histogram(array, n, bins, k);
	naive_histogram(array, n, bins2, k);
	for(i=0;i<800/k;++i)
		if(bins[i] != bins2[i])
		{
			fprintf(stderr, "%d: %d %d WRONG!!\n",i, bins[i], bins2[i]);
			break;
		}
	if(i== 800 / k)
		fprintf(stderr, "OK!\n");

	return 0;
}
