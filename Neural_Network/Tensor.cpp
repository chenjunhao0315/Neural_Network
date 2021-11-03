//
//  Tensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Tensor.hpp"

Tensor::Tensor() {
    batch = 0;
    channel = 0;
    height = 0;
    width = 0;
    size = 0;
    weight = nullptr;
    delta_weight = nullptr;
}

void Tensor::reshape(int batch_, int channel_, int height_, int width_, bool extend) {
    this->free();
    batch = batch_;
    channel = channel_;
    height = height_;
    width = width_;
    size = batch * channel * height * width;
    weight = new float [size]();
    delta_weight = (extend) ? new float [size]() : nullptr;
}

void Tensor::free() {
    OTTER_FREE_ARRAY(weight);
    OTTER_FREE_ARRAY(delta_weight);
}

Tensor::~Tensor() {
    this->free();
}

Tensor::Tensor(const Tensor &T) : Tensor() {
    if (this != &T) {
        this->reshape(T.batch, T.channel, T.height, T.width, T.delta_weight);
        
        memcpy(weight, T.weight, sizeof(float) * size);
        if (T.delta_weight)
            memcpy(delta_weight, T.delta_weight, sizeof(float) * size);
    }
}

Tensor::Tensor(Tensor &&T) {
    batch = T.batch;
    width = T.width;
    height = T.height;
    channel = T.channel;
    size = T.size;
    weight = T.weight;
    T.weight = nullptr;
    delta_weight = T.delta_weight;
    T.delta_weight = nullptr;
}

Tensor& Tensor::operator=(const Tensor &T) {
    if (this != &T) {
        if (size != T.size) {
            this->reshape(T.batch, T.channel, T.height, T.width, T.delta_weight);
        }
        
        batch = T.batch;
        width = T.width;
        height = T.height;
        channel = T.channel;
        
        memcpy(weight, T.weight, sizeof(float) * size);
        if (T.delta_weight) {
            if (!delta_weight)
                this->extend();
            memcpy(delta_weight, T.delta_weight, sizeof(float) * size);
        } else {
            OTTER_FREE_ARRAY(delta_weight);
        }
    }
    return *this;
}

Tensor& Tensor::operator=(float c) {
    fill_cpu(size, weight, c);
    return *this;
}

Tensor& Tensor::operator=(std::initializer_list<float> list) {
    if (list.size() > size) {
        fprintf(stderr, "[Tensor] Initialize out of range!\n");
        return *this;
    }
    float *ptr = weight;
    for (float element : list)
        *(ptr++) = element;
    return *this;
}

float& Tensor::operator[](int index) {
    return weight[index];
}

const float& Tensor::operator[](int index) const {
    return weight[index];
}

bool Tensor::operator==(const Tensor &T) const {
    if (!same_structure(*this, T))
        return false;
    float *src_1 = weight;
    float *src_2 = T.weight;
    for (int i = size; i--; ) {
        if (*(src_1++) != *(src_2++))
            return false;
    }
    return true;
}

Tensor& Tensor::operator+=(const Tensor &T) {
    assert(same_structure(*this, T));
    float *src = T.weight;
    float *dst = weight;
    for (int i = size; i--; )
        *(dst++) += *(src++);
    return *this;
}
Tensor& Tensor::operator-=(const Tensor &T) {
    assert(same_structure(*this, T));
    float *src = T.weight;
    float *dst = weight;
    for (int i = size; i--; )
        *(dst++) -= *(src++);
    return *this;
}

Tensor::Tensor(int batch_, int channel_, int height_, int width_) : Tensor() {
    this->reshape(batch_, channel_, height_, width_, false);

    // assign random value
    float scale = sqrt(1.0 / size);
    for (int i = size; i--; ) {
        weight[i] = randn(0.0, scale);
    }
}

Tensor::Tensor(int batch_, int channel_, int height_, int width_, float parameter) : Tensor() {
    this->reshape(batch_, channel_, height_, width_, false);

    if (parameter)
        fill(weight, weight + size, parameter);
}

void Tensor::extend() {
    delta_weight = new float [size]();
}

void Tensor::copyTo(Tensor &T) {
    memcpy(T.weight, weight, sizeof(float) * min(size, T.size));
}

Tensor::Tensor(Tensor *T) : Tensor() {
    if (T) {
        this->reshape(T->batch, T->channel, T->height, T->width, T->delta_weight);
        
        memcpy(weight, T->weight, sizeof(float) * size);
        if (T->delta_weight)
            memcpy(delta_weight, T->delta_weight, sizeof(float) * size);
    }
}

Tensor::Tensor(vfloat &V) : Tensor() {
    this->reshape(1, (int)V.size(), 1, 1, false);
    
    for (int i = 0; i < size; ++i) {
        weight[i] = V[i];
    }
}

Tensor::Tensor(vfloat V1, vfloat V2, vfloat V3, int width_, int height_) : Tensor() {
    this->reshape(1, 3, height_, width_, false);
    int n = width * height;
    
    for (int i = 0; i < n; ++i) {
        weight[i] = V1[i];
        weight[i + n] = V2[i];
        weight[i + 2 * n] = V3[i];
    }
}

Tensor::Tensor(float* pixelArray, int width_, int height_, int channel_) : Tensor() {
    this->reshape(1, channel_, height_, width_, false);
    
    copy_cpu(size, pixelArray, weight);
}

void Tensor::one_of_n_encodinig(int index, int n) {
    if (size != n) {
        this->reshape(1, n, 1, 1, false);
    }
    weight[index] = 1;
}

Tensor Tensor::concate(const Tensor &T) {
    assert(same_plane(*this, T));
//    Tensor result(width, height, channel + T.channel);
    Tensor result(batch, channel + T.channel, height, width, 0);
    float *src_1 = weight;
    float *src_2 = T.weight;
    float *dst = result.weight;
    for (int i = size; i--; )
        *(dst++) = *(src_1++);
    for (int i = T.size; i--; )
        *(dst++) = *(src_2++);
    return result;
}

float* Tensor::getWeight() {
    return weight;
}

float* Tensor::getDeltaWeight() {
    return delta_weight;
}

void Tensor::showWeight() {
    int n = size;
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", weight[i]);
    }
    printf("\n");
}

void Tensor::showDeltaWeight() {
    if (delta_weight) {
        for (int i = 0; i < size; ++i) {
            printf("%.2f ", delta_weight[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "[Tensor] Delta weight doesn't exist!\n");
    }
}

void Tensor::clearWeight() {
    fill_cpu(size, weight, 0);
}

void Tensor::clearDeltaWeight() {
    fill_cpu(size, delta_weight, 0);
}

int Tensor::getWidth() {
    return width;
}

int Tensor::getHeight() {
    return height;
}

int Tensor::getDimension() {
    return channel;
}

void Tensor::shape() {
    printf("width: %d height: %d channel: %d size: %d\n", width, height, channel, size);
}

float Tensor::get(int batch_, int channel_, int height_, int width_) {
    return weight[((height_ * width) + width_) + (width * height * channel_) + (width * height * channel * batch_)];
}

void Tensor::save_raw(FILE *f) {
    fwrite(weight, sizeof(float), size, f);
}

void Tensor::load_raw(FILE *f) {
    size_t check = fread(weight, sizeof(float), size, f);
    if (check != size) {
        fprintf(stderr, "[Tensor] Unexpected end!\n");
    }
}

vfloat Tensor::toVector() {
    vfloat result;
    result.reserve(size);
    for (int i = 0; i < size; ++i) {
        result.push_back(weight[i]);
    }
    return result;
}

void Tensor::toIMG(const char *filename) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    unsigned char pixel[3 * width * height];
    for (int i = 0; i < width * height; ++i) {
        for (int j = 0; j < 3; ++j) {
            pixel[i + j] = (unsigned char)(weight[i + j * width * height] * 255);
        }
    }
    fwrite(pixel, sizeof(unsigned char), 3 * width * height, f);
    fclose(f);
}

ostream& operator<<(ostream& os, Tensor& t) {
    for (int b = 0; b < t.batch; ++b) {
        for (int c = 0; c < t.channel; ++c) {
            os << "Dim: " << c << std::endl;
            for (int h = 0; h < t.height; ++h) {
                for (int w = 0; w < t.width; ++w) {
                    os << std::fixed << std::setprecision(2) << (t.get(b, c, h, w)) << " ";
                }
                os << std::endl;
            }
        }
    }
    return os;
}

bool same_plane(const Tensor &a, const Tensor &b) {
    return a.width == b.width && a.height == b.height;
}

bool same_block(const Tensor &a, const Tensor &b) {
    return same_plane(a, b) && a.channel == b.channel;
}

bool same_structure(const Tensor &a, const Tensor &b) {
    return same_block(a, b) && a.batch == b.batch;
}

Tensor operator+(const Tensor &a, const Tensor &b) {
    assert(same_plane(a, b));
    Tensor c(a.batch, a.channel, a.height, a.width, 0);
    float *src_1 = a.weight;
    float *src_2 = b.weight;
    float *dst = c.weight;
    for (int i = a.size; i--; )
        *(dst++) = *(src_1++) + *(src_2++);
    
    return c;
}

Tensor operator-(const Tensor &a, const Tensor &b) {
    assert(same_plane(a, b));
    Tensor c(a.batch, a.channel, a.height, a.width, 0);
    float *src_1 = a.weight;
    float *src_2 = b.weight;
    float *dst = c.weight;
    for (int i = a.size; i--; )
        *(dst++) = *(src_1++) - *(src_2++);
    
    return c;
}
