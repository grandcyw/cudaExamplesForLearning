#include <iostream>
#include <vector>
#include <unistd.h>
#include <malloc.h>

void check_memory() {
    std::cout << "按回车查看内存变化...";
    std::cin.get();
}

int main() {
    check_memory(); // 基准内存

    int* ptr = new int[1000000]; // 分配约4MB
    check_memory(); // 内存上升

    delete[] ptr; 
    check_memory(); // 内存不降（保留在内存池）

    // 强制返还OS
    malloc_trim(0); // Linux特有
    check_memory(); // 内存下降
}