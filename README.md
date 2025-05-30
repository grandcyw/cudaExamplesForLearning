-- nvcc -rdc=true  merge_sort.cu -o merge_sort -lcudadevrt启用动态并行在核函数中发射子kernel
-- makefile亦可得到可执行文件
高性能排序用thrust库或者cub库的radix_sort或者bitonic
纯手写kernel
