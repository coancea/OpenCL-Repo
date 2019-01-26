# OpenCL-Repo
Various code examples in OpenCL

## Tentative schedule

### DAY 1

I. Lecture 9:00 - 11:00:

1. Hardware Trends motivating : power wall, memory/bandwidth wall

2. GPU vs CPU architecture. GPUs (AMD) centered around three key ideas:
    a) transistors used for massive parallelism rather than for programming
        convenience (caches + control units).
    b) SIMD: amortize cost/complexity of managing an instruction stream
        across many ALUs;
    c) Hardware multi threading for hiding memory latency.
    d) capricious hardware to program, e.g., divergence + coalescing

II. Guided programming exercises (11:00 - 12:00; 13:00 - 5:00pm):

Introduction to OpenCL programming model
- buffers + CPU-GPU transfer + control queues + enqueueing kernel +
events, etc.):

Fill in the blank exercises:

* simple hello world

* example demonstrating thread divergence (load balancing) issues

* 2-d stencil exercising texture memory + 2-d kernel

* demonstrating profiling + debugging + printing

* maybe naive matrix-matrix multiplication

### DAY 2

I. Lecture 9:00 - 12:00:

1. Simple dependency analysis on arrays; when is a loop parallel and
when is it safe to interchange or distribute a loop?

2. Optimizing temporal locality by tiling demonstrated on
   matrix-matrix multiplication (MMM):
    * Block tiling as loop strip-mining + loop interchange
    * starting from the naive MMM we derive a block-tiled version and

   a block+register tiled version (in C pseudocode)
    * GPU: local memory + barrier synchronization

3. Optimizing spatial locality:
    * what are coalesced accesses to global memory
    * transforming coalesced to uncoalesced accesses by transposition
    * how to implement a transposition kernel in which all read/write
      accesses are

II. Guided programming exercises (13:00 - 5:00pm):
Walking over the provided code that aims to demonstrate the topics
covered in lecture.

Fill In the blanks exercise:
    -- the register + block tiled version of matrix-matrix multiplication
    -- optimize a (contrived) program to have only coalesced accesses
to global memory by means of transposition.
    -- then optimize it further by fusing the transposition inside the
program.

### DAY 3

I. Lecture 9:00 - 12:00:

Data parallel building blocks: map, reduce, (segmented) scan semantics
and GPU implementation ideas.
Data-parallel thinking: compose programs like puzzles from a nested
composition of such bulk operators.
Main optimization: fusion.
Applications: maximal-segment sum problem (MSSP), two-way
partitioning, sparse-matrix-vector multiplication.

II. Guided programming exercises (11:00 - 12:00; 13:00 - 5:00pm):

Take a look at the implementation of reduce/scan.

Fill in the blanks exercise:
    * map-reduce fusion for MSSP.
    * optimizing two-way partitioning an array (scan-scan horizontal
fusion; map-scan fusion).
    * sparse-matrix vector multiplication

### DAY 4

I. Lecture 9:00 - 12:00:

Histogram formulation and "reduce-by-key" generalization;
    optimization space and possible implementation strategies

Stencil fusion (\cite{Halide})

Streaming: overlapping communication + computation.

II. Guided programming exercises (13:00 - 5:00pm):
* "atomics" support in OpenCL (atomic-add, CAS, compare-and-exchange)
* walking over the provided code that aims to demonstrate the topics covered in lecture.
* fill in the blank exercises regarding histogram implementation
* fill in the blank exercise regarding stencil fusion.
* demonstrating overlapping communication + computation

### DAY 5 --

(4 hours)

Practical considerations: here we discuss computational kernels of
interest to BkMedical (proposed by Franck)
By discuss, I mean fill in the blank exercises through which we guide
the audience to develop and efficient solution.

## Stuff for Troels to do

* GPU architecture/motivation
* GPU-specific details (AMD)
* Hello-world kernels
* Command queue/events (streaming?)
* 2D stencil, with texture memory
* Profiling and debugging
* Naive matrix-matrix multiplication
* Reduction

### Exercise ideas

* MSSP (map/reduce fusion)
* Parallel reduction of matrix rows
* Sparse-Matrix Vector multiplication
