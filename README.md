# CUDA Merge Sort Implementation

![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

高性能并行排序方案，支持动态并行(Dynamic Parallelism)和多种优化策略。

## 特性

- ✅ **双模式支持**
  - 纯手写Merge Sort Kernel（教学用途）
  - Thrust/CUB高性能排序（生产环境推荐）
- 🚀 **动态并行**：支持核函数内启动子kernel (`-rdc=true`)
- 📊 **性能对比**：提供与CPU std::sort的基准测试

## 编译选项

```bash
# 基础编译（无动态并行）
nvcc merge_sort.cu -o merge_sort

# 启用动态并行
nvcc -rdc=true merge_sort.cu -o merge_sort -lcudadevrt

# 使用Makefile
make        # 编译默认版本
make dp=1   # 启用动态并行
