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
                variance_value += pow((*(src++) - mean_value), 2);
            }
        }
    }
    scal_cpu(dimension, scale, variance);
}


void normalize(float *src, float *mean, float *variance, int batch_size, int dimension, int size) {
    float mean_value, variance_scale;
    
    for (int b = batch_size; b--; ) {
        for (int d = 0; d < dimension; ++d) {
            mean_value = mean[d];
            variance_scale = 1.0 / (sqrt(variance[d] + 0.000001f));
            for (int i = size; i--; ++src) {
                *src = (*(src) - mean_value) * variance_scale;
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
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + 0.000001));
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

void scal_add_cpu(int size, float scale, float bias, float *src) {
    for (int i = size; i--; ++src)
        *(src) = *(src) * scale + bias;
}

void axpy_cpu(int size, float scale, float *src, float *dst) {
    for (int i = size; i--; )
        *(dst++) += scale * *(src++);
}

void fill_cpu(int size, float *src, float parameter) {
    for (int i = size; i--; )
        *(src++) = parameter;
}

void mul_cpu(int size, float *src1, float *src2, float *dst) {
    for (int i = size; i--; )
        *(dst++) = *(src1++) * *(src2++);
}

void div_cpu(int size, float *src1, float *src2, float *dst) {
    for (int i = size; i--; )
        *(dst++) = *(src1++) / *(src2++);
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
    switch (method) {
        case LOGISTIC:
            return logistic_activate(src);
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

float constrain(float min, float max, float a) {
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc) {
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(int i = 0; i < M; ++i) {
        for(int k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for(int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(int i = 0; i < M; ++i){
        for(int k = 0; k < K; ++k){
            float A_PART = ALPHA * A[k * lda + i];
            for(int j = 0; j < N; ++j){
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc) {
    if (BETA != 1) {
        for(int i = 0; i < M; ++i){
            for(int j = 0; j < N; ++j){
                C[i * ldc + j] *= BETA;
            }
        }
    }
    if(!TA && !TB)      // A, B not transpose
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB) // A transpose, B not
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB) // B transpose, A not
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else                // A, B transpose
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    
//    #pragma omp parallel for num_threads(OMP_THREADS)
//    for (int t = 0; t < M; ++t) {
//        if (!TA && !TB)
//            gemm_nn(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
//        else if (TA && !TB)
//            gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
//        else if (!TA && TB)
//            gemm_nt(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
//        else
//            gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
//    }
}

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;
    return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im, int channels, int height, int width, int ksize,  int stride, int pad, float* data_col) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;

    const int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad, float val) {
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

void col2im_cpu(float* data_col, int channels,  int height,  int width, int ksize,  int stride, int pad, float* data_im) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad, val);
            }
        }
    }
}

inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)(a) < (unsigned)(b);
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    int channel, kernel_row, kernel_col, output_rows, output_col;
    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_col = output_w; output_col; output_col--) {
                            *(data_col++) = 0;
                        }
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
void col2im_cpu_ext(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) {
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    int channel, kernel_row, kernel_col, output_rows, output_col;
    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        data_col += output_w;
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}
