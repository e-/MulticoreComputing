bfs: bfs.c
	gcc -std=c99 -DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT -I.././generator bfs.c ../graph500.c ../options.c ../rmat.c ../kronecker.c ../verify.c ../prng.c ../xalloc.c ../timer.c ../generator/splittable_mrg.c ../generator/graph_generator.c ../generator/make_graph.c ../generator/utils.c  -lm -lrt -o bfs -lOpenCL
