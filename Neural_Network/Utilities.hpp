//
//  Utilities.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Utilities_hpp
#define Utilities_hpp

#include <stdio.h>
#include <random>
#include <chrono>
#include <vector>
#include "Tensor.hpp"
#include "Box.hpp"
#include "omp.h"

#define OMP_THREADS 4

#define OTTER_FREE(data) if (data) delete data; data = nullptr;
#define OTTER_FREE_ARRAY(data) if (data) delete [] data; data = nullptr;
#define OTTER_CHECK_PTR_QUIT(ptr, message, error_code)   \
    if (!ptr) {fprintf(stderr, message); exit(error_code);}
#define OTTER_CHECK_PTR_BOOL(ptr, message)   \
    if (!ptr) {fprintf(stderr, message); return false;}

using namespace std::chrono;
using std::vector;

typedef vector<float> vfloat;

static std::random_device rd;
static std::mt19937 generator(rd());
static std::uniform_real_distribution<float> unif(-1.0, 1.0);

float Random();
float Random(float min, float max);
float Random_scale(float s);
float Random_precal(float min, float max, float rand);
float guassRandom();
float randn(float mu, float std);


void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void cal_mean(float *src, int batch_size, int dimension, int size, float *mean);
void cal_variance(float *src, float *mean, int batch_size, int dimension, int size, float *variance);
void normalize(float *src, float *mean, float *variance, int batch_size, int dimension, int size);
void fill_cpu(int size, float *src, float parameter);
void copy_cpu(int size, float *src, float *dst);
void scal_cpu(int size, float scale, float *src);
void scal_add_cpu(int size, float scale, float bias, float *src);
void axpy_cpu(int size, float scale, float *src, float *dst);
void mul_cpu(int size, float *src1, float *src2, float *dst);
void div_cpu(int size, float *src1, float *src2, float *dst);
void sub_cpu(int size, float mean, float *src);

// ACTIVATION
enum ACTIVATE_METHOD {
    LOGISTIC
};

float sum_array(float *a, int n);
void activate_array(float *src, int length, ACTIVATE_METHOD method);

float activate(float src, ACTIVATE_METHOD method);
static inline float logistic_activate(float x) {return 1.f / (1.f + expf(-x));}
static inline float tanh_activate(float x) {return (2 / (1 + expf(-2 * x)) - 1); }
static inline float softplus_activate(float x, float threshold) {
    if (x > threshold) return x;
    else if (x < -threshold) return expf(x);
    return logf(expf(x) + 1);
}
// END ACTIVATION

float constrain(float min, float max, float a);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

inline static int is_a_ge_zero_and_a_lt_b(int a, int b);
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col);
void col2im_cpu_ext(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im);

void convert_index_base_to_channel_base(float *src, float *dst, int w, int h, int c);
template <typename srcType = float, typename dstType = float>
void convert_channel_base_to_index_base(void *src, void *dst, int w, int h, int c) {
    srcType *src_ptr = (srcType*)src, *channel_ptr = (srcType*)src;
    dstType *dst_ptr = (dstType*)dst_ptr;
}

class Clock {
public:
    Clock();
    void start();
    void stop();
    
    template <typename T = milliseconds>
    long long int getElapsed();
    template <typename T = milliseconds>
    void stop_and_show(const char *postfix = "ms");
    template <typename T = milliseconds>
    void lap_and_show(const char *postfix = "ms");
    template <typename T = milliseconds>
    void show(const char *postfix = "ms");
private:
    high_resolution_clock::time_point time_start;
    high_resolution_clock::time_point time_stop;
};

template <typename T>
void Clock::stop_and_show(const char *postfix) {
    time_stop = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_stop - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
void Clock::lap_and_show(const char *postfix) {
    auto time_lap = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_lap - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
void Clock::show(const char *postfix) {
    auto duration = duration_cast<T>(time_stop - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
long long int Clock::getElapsed() {
    auto time_lap = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_lap - time_start);
    return duration.count();
}

#endif /* Utilities_hpp */
