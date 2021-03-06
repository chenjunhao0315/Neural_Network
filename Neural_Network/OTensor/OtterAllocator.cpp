//
//  OtterAllocator.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/12/9.
//

#include "OtterAllocator.hpp"

void deleteNothing(void*) {}

Allocator* get_default_allocator() {
    return &default_allocator;
}

Allocator* GetAllocator(Device device) {
    switch (device) {
        case Device::CPU: get_default_allocator(); break;
    }
    return get_default_allocator();
}

void *otter_malloc_log(const size_t size, const char * const filename, const char * const funcname, const int line) {
    if (size == 0) return nullptr;
    void *ptr = malloc(size);
    if (!ptr) fprintf(stderr, "Failed to malloc %zu(bytes) at File: %s Func: %s Line: %d\n", size, filename, funcname, line);
    return ptr;
}

void *otter_calloc_log(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line) {
    if (size == 0 || nmemb == 0) return 0;
    void *ptr = calloc(nmemb, size);
    if (!ptr) fprintf(stderr, "Failed to calloc %zu(bytes) at File: %s Func: %s Line: %d\n", nmemb * size, filename, funcname, line);
    return ptr;
}

void *otter_realloc_log(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line) {
    ptr = realloc(ptr,size);
    if (!ptr) fprintf(stderr, "Failed to realloc %zu(bytes) at File: %s Func: %s Line: %d\n", size, filename, funcname, line);
    return ptr;
}

void otter_free(void *ptr) {
    free(ptr);
    ptr = nullptr;
}
