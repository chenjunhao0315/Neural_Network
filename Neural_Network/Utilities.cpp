//
//  Utilities.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Utilities.hpp"

static bool return_v = false;
static float v_val = 0.0;

float randn(float mu, float std) {
    return mu + guassRandom() * std;
}

float guassRandom() {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    float u = Random();
    float v = Random();
    float r = u * u + v * v;
    if (r == 0 || r > 1)
        return guassRandom();
    float c = sqrt(-2 * log(r) / r);
    v_val = v * c;
    return_v = true;
    return u * c;
}

float Random() {
    return unif(generator);
}

Clock::Clock() {
    time_start = high_resolution_clock::now();
    time_stop = high_resolution_clock::now();
}

void Clock::start() {
    time_start = high_resolution_clock::now();
}

void Clock::stop() {
    time_stop = high_resolution_clock::now();
}

void cal_mean(float *src, int batch_size, int dimension, int size, int *input_index, float *mean) {
    float scale = 1.0 / (batch_size * size);
    int *index = input_index, *index_ptr;
    int one_batch_size = size * dimension;
    int d, b, i;

    fill_cpu(dimension, mean, 0);
    for (b = 0; b < batch_size; ++b) {
        index_ptr = index;
        for (d = 0; d < dimension; ++d) {
            float &mean_value = mean[d];
            for (i = 0; i < size; ++i) {
                mean_value += src[*(index_ptr++)];
            }
        }
        src += one_batch_size;
    }
    scal_cpu(dimension, scale, mean);
}

void cal_variance(float *src, float *mean, int batch_size, int dimension, int size, int *input_index, float *variance) {
    float scale = 1.0 / (batch_size * size - 1);
    int *index = input_index, *index_ptr;
    int one_batch_size = dimension * size;
    int d, b, i;
    float mean_value;

    fill_cpu(dimension, variance, 0);
    for (b = batch_size; b--; ) {
        index_ptr = index;
        for (d = 0; d < dimension; ++d) {
            float &variance_value = variance[d];
            mean_value = mean[d];
            for (i = 0; i < size; ++i) {
                variance_value += pow((src[*(index_ptr++)] - mean[d]), 2);
            }
        }
        src += one_batch_size;
    }
    scal_cpu(dimension, scale, variance);
}

void normalize(float *src, float *mean, float *variance, int batch_size, int dimension, int size, int *input_index) {
    int *index_ptr, index;
    int one_batch_size = dimension * size;
    float mean_value, variance_scale;
    
    for (int b = batch_size; b--; ) {
        index_ptr = input_index;
        for (int d = 0; d < dimension; ++d) {
            mean_value = mean[d];
            variance_scale = 1.0 / (sqrt(variance[d]) + 0.000001f);
            for (int i = size; i--; ) {
                index = *(index_ptr++);
                src[index] = (src[index] - mean_value) * variance_scale;
            }
        }
        src += one_batch_size;
    }
}

void copy_cpu(int size, float *src, float *dst) {
    for (int i = size; i--; )
        *(dst++) = *(src++);
}

void scal_cpu(int size, float scale, float *src) {
    for (int i = size; i--; )
        *(src++) *= scale;
}

void axpy_cpu(int size, float scale, float *src, float *dst) {
    for (int i = size; i--; )
        *(dst++) += scale * *(src++);
}

void fill_cpu(int size, float *src, float parameter) {
    for (int i = size; i--; )
        *(src++) = parameter;
}
