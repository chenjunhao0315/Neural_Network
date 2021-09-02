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

float Random(float min, float max) {
    return (Random() + 1) / 2.0 * (max - min) + min;
}

float Random_scale(float s) {
    float scale = Random(1, s);
    if(Random() > 0)
        return scale;
    return 1.0 / scale;
}

float Random_precal(float min, float max, float rand) {
    if (min > max)
        std::swap(min, max);
    return (rand * (max - min)) + min;
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

void cal_mean(float *src, int batch_size, int dimension, int size, float *mean) {
    float scale = 1.0 / (batch_size * size);
    int d, b, i;

    fill_cpu(dimension, mean, 0);
    for (b = 0; b < batch_size; ++b) {
        for (d = 0; d < dimension; ++d) {
            float &mean_value = mean[d];
            for (i = 0; i < size; ++i) {
                mean_value += *(src++);
            }
            if (isnan(mean_value) || isinf(mean_value))
                printf("strange\n");
        }
    }
    scal_cpu(dimension, scale, mean);
}

void cal_variance(float *src, float *mean, int batch_size, int dimension, int size, float *variance) {
    float scale = 1.0 / (batch_size * size - 1);
    int d, b, i;
    float mean_value;

    fill_cpu(dimension, variance, 0);
    for (b = batch_size; b--; ) {
        for (d = 0; d < dimension; ++d) {
            float &variance_value = variance[d];
            mean_value = mean[d];
            for (i = 0; i < size; ++i) {
                variance_value += pow((*(src++) - mean[d]), 2);
            }
            if (variance_value == 0)
                printf("strange\n");
        }
    }
    scal_cpu(dimension, scale, variance);
}


void normalize(float *src, float *mean, float *variance, int batch_size, int dimension, int size) {
    float mean_value, variance_scale;
    
    for (int b = batch_size; b--; ) {
        for (int d = 0; d < dimension; ++d) {
            mean_value = mean[d];
            variance_scale = 1.0 / (sqrt(variance[d]) + 0.000001f);
            if (isnan(variance_scale) || isinf(variance_scale))
                printf("error\n");
            for (int i = size; i--; ) {
                if (isnan(*src) || isinf(*src))
                    printf("error\n");
                *src = (*(src) - mean_value) * variance_scale;
                if (isnan(*src) || isinf(*src))
                    printf("error\n");
                ++src;
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean) {
    float scale = 1.0 / (batch * spatial);
    for(int i = 0; i < filters; ++i){
        mean[i] = 0;
        for(int j = 0; j < batch; ++j){
            for(int k = 0; k < spatial; ++k){
                int index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance) {
    float scale = 1.0 / (batch * spatial - 1);
    for(int i = 0; i < filters; ++i){
        variance[i] = 0;
        for(int j = 0; j < batch; ++j){
            for(int k = 0; k < spatial; ++k){
                int index = j * filters * spatial + i * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial) {
    for(int b = 0; b < batch; ++b){
        for(int f = 0; f < filters; ++f){
            for(int i = 0; i < spatial; ++i){
                int index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + 0.000001);
            }
        }
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

float sum_array(float *a, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i) sum += a[i];
    return sum;
}

void activate_array(float *src, int length, ACTIVATE_METHOD method) {
    for(int i = 0; i < length; ++i){
        src[i] = activate(src[i], method);
    }
}

float activate(float src, ACTIVATE_METHOD method) {
    if (isnan(src))
        printf("error\n");
    switch (method) {
        case SIGMOID:
            return 1.0 / (1.0 + exp(-src));
        default:
            break;
    }
    return 0;
}

void convert_index_base_to_channel_base(float *src, float *dst, int w, int h, int c) {
    int channel_size = w * h;
    float *src_ptr = src, *channel_ptr = src;
    for (int d = 0; d < c; ++d) {
        src_ptr = channel_ptr;
        for (int i = channel_size; i--; ) {
            *(dst++) = *(src_ptr);
            src_ptr += c;
        }
        ++channel_ptr;
    }
}

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Box &a, Box &b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0)
        return 0;
    float area = w*h;
    return area;
}

float box_union(Box &a, Box &b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(Box &a, Box &b) {
    return box_intersection(a, b) / box_union(a, b);
}

Box vfloat_to_box(vfloat &src, int index) {
    Box b;
    b.x = src[index + 0];
    b.y = src[index + 1];
    b.w = src[index + 2];
    b.h = src[index + 3];
    return b;
}

float constrain(float min, float max, float a) {
    if (a < min) return min;
    if (a > max) return max;
    return a;
}
