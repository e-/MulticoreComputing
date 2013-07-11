/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010,  Georgia Institute of Technology, USA. */
/* See COPYING for license. */
#define _FILE_OFFSET_BITS 64
#define _THREAD_SAFE

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <assert.h>
#include <pthread.h>

#include "../compat.h"
#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"

#define MINVECT_SIZE 2
#define GPU_LOCAL_SIZE 64
#define CPU_LOCAL_SIZE 1024

#define MAX_DEV 50

#define CPU_WEIGHT 12
#define GPU_WEIGHT 1

static int maxvtx, nv, sz;
static int * restrict xoff; /* Length 2*nv+2 */
static int * restrict xadjstore; /* Length MINVECT_SIZE + (xoff[nv] == nedge) */
static int * restrict xadj;
cl_platform_id	platform;
cl_device_id	device[MAX_DEV];
cl_context	context;
cl_command_queue	command_queue[MAX_DEV];
cl_program	program;
cl_kernel	kernel1[MAX_DEV], kernel2[MAX_DEV];

cl_mem	xadj_buffer[MAX_DEV], 
		xoff_buffer[MAX_DEV],
		state_buffer[MAX_DEV], 
		bfs_tree_buffer[MAX_DEV],
		finished_buffer[MAX_DEV];
char *global_state, *finished, *state[MAX_DEV];
int *bfs_trees[MAX_DEV];
int *global_bfs_tree;
size_t xadj_buffer_size[MAX_DEV], xoff_buffer_size[MAX_DEV], global_state_buffer_size, global_bfs_tree_buffer_size, finished_buffer_size;

int weight[MAX_DEV], start[MAX_DEV], end[MAX_DEV], xadj_offset[MAX_DEV];
int	ndev;
size_t global[MAX_DEV][1], local[MAX_DEV][1];
int64_t * restrict bfs_tree;

const char * kernel_src = 
"__kernel void step1(__global const int *xadj,__global const int *xoff, __global char *state, __global int *bfs_tree, const int maxvtx, const int offset, const int xadj_offset){" 
"	const int tid = get_global_id(0);"
"	const int real_id = tid + offset;"
"	if(real_id <= maxvtx && state[real_id] == 2){"
"		state[real_id] = 3;"
"		const int end = xoff[1+2*tid] - xadj_offset;"
"		for(int i=xoff[2*tid] - xadj_offset;i<end;++i){"
"			const int j = xadj[i];"
"			if(state[j] == 0){"
"				state[j] = 1;"
"				bfs_tree[j] = real_id;"
"			}"
"		}"
"	}"
"}"
"__kernel void step2(__global char *state, __global char *finished, const int maxvtx, const int offset){"
"	const int tid = get_global_id(0) + offset;"
"	if(tid <= maxvtx && state[tid] == 1) {"
"		state[tid] = 2;"
"		*finished = 0;"
"	}"
"}"
;

static void
find_nv (const struct packed_edge * restrict IJ, const int64_t nedge)
{
	int k;

	maxvtx = -1;
	for (k = 0; k < nedge; ++k) {
	  if (get_v0_from_edge(&IJ[k]) > maxvtx)
	    maxvtx = get_v0_from_edge(&IJ[k]);
	  if (get_v1_from_edge(&IJ[k]) > maxvtx)
	    maxvtx = get_v1_from_edge(&IJ[k]);
	}
	nv = 1+maxvtx;
}

static int
alloc_graph (int nedge)
{
	sz = (2*nv+2) * sizeof (*xoff);
	xoff = (int *)xmalloc_large_ext (sz);
	if (!xoff) return -1;
	return 0;
}

static void
free_graph (void)
{
	xfree_large (xadjstore);
	xfree_large (xoff);
}

#define XOFF(k) (xoff[2*(k)])
#define XENDOFF(k) (xoff[1+2*(k)])

int accum;

static int
setup_deg_off (const struct packed_edge * restrict IJ, int nedge)
{
	int k;
	for (k = 0; k < 2*nv+2; ++k)
	  xoff[k] = 0;
	for (k = 0; k < nedge; ++k) {
	  int i = get_v0_from_edge(&IJ[k]);
	  int j = get_v1_from_edge(&IJ[k]);
	  if (i != j) { /* Skip self-edges. */
	    if (i >= 0) ++XOFF(i);
	    if (j >= 0) ++XOFF(j);
	  }
	}
	accum = 0;
	for (k = 0; k < nv; ++k) {
	  int tmp = XOFF(k);
	  if (tmp < MINVECT_SIZE) tmp = MINVECT_SIZE;
	  XOFF(k) = accum;
	  accum += tmp;
	}
	XOFF(nv) = accum;
	for (k = 0; k < nv; ++k)
	  XENDOFF(k) = XOFF(k);
	if (!(xadjstore = (int *)xmalloc_large_ext ((accum + MINVECT_SIZE) * sizeof (*xadjstore))))
	  return -1;
	xadj = &xadjstore[MINVECT_SIZE]; /* Cheat and permit xadj[-1] to work. */
	for (k = 0; k < accum + MINVECT_SIZE; ++k)
	  xadjstore[k] = -1;
	return 0;
}

static void
scatter_edge (const int i, const int j)
{
	int where;
	where = XENDOFF(i)++;
	xadj[where] = j;
}

static int
i64cmp (const void *a, const void *b)
{
	const int ia = *(const int*)a;
	const int ib = *(const int*)b;
	if (ia < ib) return -1;
	if (ia > ib) return 1;
	return 0;
}

static void
pack_vtx_edges (const int64_t i)
{
	int64_t kcur, k;
	if (XOFF(i)+1 >= XENDOFF(i)) return;
	qsort (&xadj[XOFF(i)], XENDOFF(i)-XOFF(i), sizeof(*xadj), i64cmp);
	kcur = XOFF(i);
	for (k = XOFF(i)+1; k < XENDOFF(i); ++k)
	  if (xadj[k] != xadj[kcur])
	    xadj[++kcur] = xadj[k];
	++kcur;
	for (k = kcur; k < XENDOFF(i); ++k)
	  xadj[k] = -1;
	XENDOFF(i) = kcur;
}

static void
pack_edges (void)
{
	int64_t v;

	for (v = 0; v < nv; ++v)
	  pack_vtx_edges (v);
}

static void
gather_edges (const struct packed_edge * restrict IJ, int64_t nedge)
{
	int64_t k;

	for (k = 0; k < nedge; ++k) {
	  int64_t i = get_v0_from_edge(&IJ[k]);
	  int64_t j = get_v1_from_edge(&IJ[k]);
	  if (i >= 0 && j >= 0 && i != j) {
	    scatter_edge (i, j);
	    scatter_edge (j, i);
	  }
	}

	pack_edges ();
}

int 
create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge)
{
	int i, err;
	size_t kernel_src_len = strlen(kernel_src);

	find_nv (IJ, nedge); //matvtx에 최대값 넣음 
	if (alloc_graph (nedge)) return -1;
	if (setup_deg_off (IJ, nedge)) {
	  xfree_large (xoff);
	  return -1;
	}
	gather_edges (IJ, nedge);
	fprintf(stderr, "XADJ: %ld\n", sizeof(xadj));

	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, (unsigned int*)&ndev);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ndev, device, NULL);
	fprintf(stderr, "ndev = %d\n", ndev);

	context = clCreateContext(0, ndev, device, NULL, NULL, NULL);
	program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, &err);
	if(err == CL_SUCCESS)
	  fprintf(stderr, "program created!\n");
	else {
	  fprintf(stderr, "program create failed!\n");
	  fprintf(stderr, "ERROR = %d, %d, %d, %d, %d, %d\n", err, 
	  CL_INVALID_CONTEXT ,
	  CL_INVALID_VALUE, 
	  CL_OUT_OF_HOST_MEMORY ,
	CL_OUT_OF_HOST_MEMORY ,
	CL_OUT_OF_HOST_MEMORY 
	);
	}
	err = clBuildProgram(program, ndev, device, NULL, NULL, NULL);
	if(err == CL_SUCCESS)
	  fprintf(stderr, "program built!\n");
	else {
		fprintf(stderr, "program built failed!\n");
		char log[5000];
		err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
		fprintf(stderr, "ERROR = %d, %s, %d, %d, %d\n", err, log,
		CL_INVALID_DEVICE ,
		CL_INVALID_VALUE ,
		CL_INVALID_PROGRAM 
		);
	}



	int maxvtx_adjusted = maxvtx;
	if(maxvtx % CPU_LOCAL_SIZE > 0){
	  maxvtx_adjusted = (maxvtx / CPU_LOCAL_SIZE + 1) * CPU_LOCAL_SIZE;
	}

	//maxvtx_adjusted를 할당해야함
	
	cl_device_type type;
	int total_weight = 0;
	int last_start = 0, count;

	for(i=0;i<ndev;++i){
		clGetDeviceInfo(device[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
		if(type == CL_DEVICE_TYPE_CPU) {
			fprintf(stderr, "CPU detected!\n");
			weight[i] = CPU_WEIGHT;
			local[i][0] = CPU_LOCAL_SIZE;
		}
		else if(type == CL_DEVICE_TYPE_GPU) {
			fprintf(stderr, "GPU detected!\n");
			weight[i] = GPU_WEIGHT;
			local[i][0] = GPU_LOCAL_SIZE;
		}
		else {
			fprintf(stderr, "UNKNOWN detected!\n");
		}
		total_weight += weight[i];
	}
	for(i=0;i<ndev;++i){
		start[i] = last_start;
		count = maxvtx_adjusted * weight[i] / total_weight;
		if(count % local[i][0] > 0)
			count = (count / local[i][0] + 1) * local[i][0];
		end[i] = last_start + count;
		global[i][0] = count;
		maxvtx_adjusted -= count;
		total_weight -= weight[i];
		last_start += count;
		fprintf(stderr, "%d: %d ~ %d\n", i, start[i], end[i]);
	}

	global_state_buffer_size = nv * sizeof(char);
	global_bfs_tree_buffer_size = nv * sizeof(int64_t);
	finished_buffer_size = sizeof(char);

	finished = (char *)xmalloc_large(nv * sizeof(char));
	global_state = (char *)xmalloc_large(global_state_buffer_size);

	for(i=0;i<ndev;++i){
		int j;
		for(j = end[i]; j>0 ;j--){
			if(XENDOFF(j) > 0){
				break;
			}
		}
		fprintf(stderr, "%d device : XOFF: %d ~ XENDOFF: %d\n", i, XOFF(start[i]), XENDOFF(j));
		bfs_trees[i] = (int *)xmalloc_large(global_bfs_tree_buffer_size);
		state[i] = (char *)xmalloc_large(global_state_buffer_size);
		xadj_buffer_size[i] = (XENDOFF(j) - XOFF(start[i])) * sizeof(*xadjstore);
		xoff_buffer_size[i] = (end[i] - start[i]) * 2 * sizeof(*xoff); 
		xadj_offset[i] = XOFF(start[i]);
		
		state_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE/* | CL_MEM_USE_HOST_PTR*/, global_state_buffer_size, NULL /* global_state */, &err);
		bfs_tree_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE/* | CL_MEM_USE_HOST_PTR*/, global_bfs_tree_buffer_size, NULL /* global_state */, &err);
		xadj_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, xadj_buffer_size[i], NULL, &err);
		xoff_buffer[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, xoff_buffer_size[i], NULL, &err);
		finished_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, finished_buffer_size, NULL, &err);
		command_queue[i] = clCreateCommandQueue(context, device[i], 0, NULL);
		
		err = clEnqueueWriteBuffer(command_queue[i], xadj_buffer[i], CL_FALSE, 0, xadj_buffer_size[i], &xadj[XOFF(start[i])], 0, NULL, NULL);
		err = clEnqueueWriteBuffer(command_queue[i], xoff_buffer[i], CL_FALSE, 0, xoff_buffer_size[i], &XOFF(start[i]), 0, NULL, NULL);
		kernel1[i] = clCreateKernel(program, "step1", &err);
		if(err == CL_SUCCESS)
		  fprintf(stderr, "kernel1 built!\n");
		else  {
			fprintf(stderr, "kernel1 error!\n");
	  		fprintf(stderr, "ERROR = %d, %d, %d, %d, %d, %d\n", err,
				CL_INVALID_PROGRAM ,
				CL_INVALID_PROGRAM_EXECUTABLE ,
				CL_INVALID_KERNEL_NAME ,
				CL_INVALID_KERNEL_DEFINITION ,
				CL_INVALID_VALUE 
			);
		}

		kernel2[i] = clCreateKernel(program, "step2", NULL);
	}
	for(i=0;i<ndev;++i)
		clFinish(command_queue[i]);
	
	return 0;
}

#define NUM_THREADS 20

int num_per_thread;

void *merge(void *t){
	int id = (long)t;
	int start, end, i, j;
	int finished = 1;
	start = num_per_thread * id;
	end = num_per_thread * (id + 1);
	if(end > nv) end=nv;
	
	for(i=start;i<end;++i){
		if(global_state[i] == 3)continue;
		for(j=0;j<ndev;++j){
			if(state[j][i] == 1 && global_state[i] != 2){
				global_state[i] = 2; //state[j][i];
				finished = 0;
				bfs_tree[i] = bfs_trees[j][i];
				break;
			}
		}
	}
	pthread_exit((void *)finished);
}

int merge_global_state(){
	pthread_t threads[NUM_THREADS];
	pthread_attr_t attr;
	int i;

	num_per_thread = nv / NUM_THREADS + 1;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(i=0;i<NUM_THREADS;++i){
		pthread_create(&threads[i], NULL, merge, (void *)i);
	}
	int rc, finished = 1;
	void * status;
	for(i=0;i<NUM_THREADS;++i){
		rc = pthread_join(threads[i], &status);
		if((long)status == 0)
			finished = 0;
	}
		
	return finished;
}

int
make_bfs_tree (int64_t *bfs_tree_out, int64_t *max_vtx_out,
	       int64_t srcvtx)
{
	bfs_tree = bfs_tree_out;
	int err = 0;
	int i, j;
	
	*max_vtx_out = maxvtx;
	bfs_tree[srcvtx] = srcvtx;

	memset(global_state, 0, global_state_buffer_size);
	for(i=0;i<ndev;++i){
		memset(state[i], 0, global_state_buffer_size);
	}

	global_state[srcvtx] = 2;

	char global_finished = 1;
	int count = 1;

	memset(bfs_tree, -1, global_bfs_tree_buffer_size);
	bfs_tree[srcvtx] = srcvtx;

	for(i=0;i<ndev;++i){
		err = clEnqueueWriteBuffer(command_queue[i], bfs_tree_buffer[i], CL_FALSE, 0, global_bfs_tree_buffer_size, bfs_tree, 0, NULL, NULL);
	}

	while(1){
		for(i=0;i<ndev;++i) {
			clEnqueueWriteBuffer(command_queue[i], state_buffer[i], CL_FALSE, 0, global_state_buffer_size, global_state, 0, NULL, NULL);
			finished[i] = 1;
			err = clEnqueueWriteBuffer(command_queue[i], finished_buffer[i], CL_FALSE, 0, finished_buffer_size, &finished[i], 0, NULL, NULL);
			
			err = clSetKernelArg(kernel1[i], 0, sizeof(cl_mem), (void *)&xadj_buffer[i]);
			err = clSetKernelArg(kernel1[i], 1, sizeof(cl_mem), (void *)&xoff_buffer[i]);
			err = clSetKernelArg(kernel1[i], 2, sizeof(cl_mem), (void *)&state_buffer[i]);
			err = clSetKernelArg(kernel1[i], 3, sizeof(cl_mem), (void *)&bfs_tree_buffer[i]);
			err = clSetKernelArg(kernel1[i], 4, sizeof(maxvtx), (void *)&maxvtx);
			err = clSetKernelArg(kernel1[i], 5, sizeof(start[i]), (void *)&start[i]);
			err = clSetKernelArg(kernel1[i], 6, sizeof(xadj_offset[i]), (void *)&xadj_offset[i]);
			err = clEnqueueNDRangeKernel(command_queue[i], kernel1[i], 1, 0, global[i], local[i], 0, 0, NULL);
			clEnqueueReadBuffer(command_queue[i], state_buffer[i], CL_FALSE, 0, global_state_buffer_size, state[i], 0, 0, 0);
			clEnqueueReadBuffer(command_queue[i], bfs_tree_buffer[i], CL_FALSE, 0, global_bfs_tree_buffer_size, bfs_trees[i], 0, 0, 0);
		}
		for(i=0;i<ndev;++i){
			clFinish(command_queue[i]);
		}
		
		global_finished = merge_global_state();
		if(global_finished)
			break;
	}

	return err;
}

void
destroy_graph (void)
{
	free_graph ();
	for(int i = 0 ; i <ndev; ++i){
		xfree_large(state[i]);
		xfree_large(bfs_trees[i]);
	}
	xfree_large(global_state);
}
