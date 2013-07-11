#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#define min(a,b) (((a) > (b)) ? (b) : (a))

int mypow2(int n){
	int c=1;
	while(n>0){c<<=1;n--;}
	return c;
}

int mylog2(int n){
	int c=0;
	while(n>1){c++;n>>=1;}
	return c;
}

void mergesort(int *array, int n)
{
	int 
		core_n = omp_get_num_procs(),
		i,
		j,
		block_size = 1,
		depth = mylog2(n),
		id,
		number_of_blocks,
		number_of_block_pairs,
		number_of_required_cores,
		number_of_block_pairs_per_thread,
		block_start1,
		block_end1,
		block_start2,
		block_end2,
		temp,
		a,
		b,
		c,
		*src = array,
		*dst,
		*ptemp,
		k;
	;

	dst = (int *)malloc(sizeof(int) * n);

	omp_set_num_threads(core_n);

	for(i = 0 ; i < depth ; ++i) 
	{
		//block_size 짜리 두개를 merge하자
		number_of_blocks = n / block_size;
		number_of_block_pairs = number_of_blocks / 2;
		number_of_required_cores = core_n;

		if(number_of_block_pairs < core_n){ //코어 숫자보다 블록 페어 수가 적으면 코어를 다 쓸필요가 없음 
			omp_set_num_threads(number_of_block_pairs);
			number_of_required_cores = number_of_block_pairs;
		}
		number_of_block_pairs_per_thread = number_of_block_pairs / number_of_required_cores;
		if(number_of_block_pairs % number_of_required_cores > 0)
			number_of_block_pairs_per_thread ++;
	
		#pragma omp parallel private(id, block_start1, block_start2, block_end1, block_end2, a, b, c, j)
		{
			id = omp_get_thread_num();
			for(j=0;j<number_of_block_pairs_per_thread;++j){
				block_start1 = id * number_of_block_pairs_per_thread * block_size * 2 + (j * 2 + 0) * block_size;
				block_end1 = block_start1 + block_size;
				block_start2 = id * number_of_block_pairs_per_thread * block_size * 2 + (j * 2 + 1) * block_size;
				block_end2 = block_start2 + block_size;

				if(block_start1 >= n)break;
				if(block_end2 >= n)block_end2 = n;

				c = block_start1;

				while(block_start1 < block_end1 && block_start2 < block_end2){
					a = src[block_start1];
					b = src[block_start2];
					
					if(a<b){ //앞에 있는것이 작으면 이걸 넣자 
						dst/*[id]*/[c++] = a;
						block_start1++;
					}
					else{ //뒤에 있는것이 작으면 이걸 넣자
						dst/*[id]*/[c++] = b;
						block_start2++;
					}
				}
				while(block_start1 < block_end1)dst/*[id]*/[c++] = src[block_start1++];
				while(block_start2 < block_end2)dst/*[id]*/[c++] = src[block_start2++];
			}
		}
		
		ptemp = src;
		src = dst;
		dst = ptemp;

		block_size <<= 1;
	}

	if(i % 2){
		//결과가 src 
		for(i=0;i<n;++i)array[i] = src[i];
		free(src);
	}else{
		free(dst);
	}
}
