#include <stdio.h>
#include <omp.h>
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
