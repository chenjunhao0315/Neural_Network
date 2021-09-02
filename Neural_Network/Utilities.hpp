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
void axpy_cpu(int size, float scale, float *src, float *dst);

void convert_index_base_to_channel_base(float *src, float *dst, int w, int h, int c);

enum ACTIVATE_METHOD {
    SIGMOID
};

float sum_array(float *a, int n);
void activate_array(float *src, int length, ACTIVATE_METHOD method);

float activate(float src, ACTIVATE_METHOD method);

float constrain(float min, float max, float a);

struct Box {
    float x, y, w, h;
};

struct Detection {
    Box bbox;
    int classes;
    vector<float> prob;
    float objectness;
    int sort_class;
};

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(Box &a, Box &b);
float box_union(Box &a, Box &b);
float box_iou(Box &a, Box &b);

Box vfloat_to_box(vfloat &src, int index);

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
