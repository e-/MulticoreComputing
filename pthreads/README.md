# pthreads

This directory contains source files which use pthread library.

## matmul.c

Multiplies two matrices using n^3 naive method.
`void matmul(float* A, float* B, float* C, int m, int n, int k, int t)` : Multiplies two matrices `A`(`m` * `k`) and `B`(`k` * `n`) using `t` threads and stores the matrix product into `C`.

### Performance

Tested on 1 node in Supercomputer Chundoong(http://chundoong.snu.ac.kr/). `t` was 32 which is the number of cores in the node.

* Multiply two
  * 5000 * 5000 matrices: 3.2s
  * 10000 * 10000 matrices: 26.3s
  * 30000 * 30000 matrices: 684s
