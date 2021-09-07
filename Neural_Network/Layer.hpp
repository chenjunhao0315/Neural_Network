//
//  Layer.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <map>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <cassert>
#include <sstream>

#include "Tensor.hpp"

using namespace std;

typedef vector<Tensor> vtensor;
typedef vector<Tensor*> vtensorptr;
typedef unordered_map<string, string> LayerOption;

struct Train_Args {
    Train_Args() {valid = false;}
    Train_Args(Tensor *kernel_, Tensor *biases_, int kernel_size_, int kernel_list_size_, int biases_size_, vfloat ln_decay_list_) : valid(true), kernel(kernel_), biases(biases_), kernel_size(kernel_size_), kernel_list_size(kernel_list_size_), biases_size(biases_size_), ln_decay_list(ln_decay_list_) {}
    bool valid;
    Tensor *kernel;
    Tensor *biases;
    int kernel_size;
    int kernel_list_size;
    int biases_size;
    vfloat ln_decay_list;
};

struct Forward_Args {
    Forward_Args(bool train_ = false, float *workspace_ = nullptr) : train(train_), workspace(workspace_) {}
    bool train;
    float *workspace;
};

static Forward_Args default_forward_args;

enum LayerType {
    Input,
    Fullyconnected,
    Relu,
    PRelu,
    LRelu,
    Softmax,
    Convolution,
    Pooling,
    EuclideanLoss,
    ShortCut,
    Sigmoid,
    BatchNormalization,
    UpSample,
    Concat,
    YOLOv3,
    Error
};

class InputLayer;
class FullyConnectedLayer;
class ReluLayer;
class PReluLayer;
class LReluLayer;
class SoftmaxLayer;
class ConvolutionLayer;
class PoolingLayer;
class EuclideanLossLayer;
class ShortCutLayer;
class SigmoidLayer;
class BatchNormalizationlayer;
class UpSampleLayer;
class ConcatLayer;
class YOLOv3Layer;

// Top layer
class Model_Layer {
public:
    ~Model_Layer();
    Model_Layer();
    Model_Layer(const Model_Layer &L);
    Model_Layer(Model_Layer &&L);
    Model_Layer& operator=(const Model_Layer &L);
    Model_Layer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* shortcut_tensor_ = nullptr, Forward_Args *args = &default_forward_args);
    void Forward(Tensor* input_tensor_, Forward_Args *args = &default_forward_args);
    float Backward(vfloat& target);
    void Backward();
    void ClearGrad();
    void shape();
    int getParameter(int type_);
    LayerType getType() {return type;}
    LayerType string_to_type(string type);
    bool save(FILE *f);
    bool load(FILE *f);
    bool load_raw(FILE *f);
    Train_Args getTrainArgs();
    int getWorkspaceSize();
private:
    LayerType type;
    InputLayer *input_layer;
    FullyConnectedLayer *fullyconnected_layer;
    ReluLayer *relu_layer;
    PReluLayer *prelu_layer;
    SoftmaxLayer *softmax_layer;
    ConvolutionLayer *convolution_layer;
    PoolingLayer *pooling_layer;
    EuclideanLossLayer *euclideanloss_layer;
    ShortCutLayer *shortcut_layer;
    LReluLayer *lrelu_layer;
    SigmoidLayer *sigmoid_layer;
    BatchNormalizationlayer *batchnorm_layer;
    UpSampleLayer *upsample_layer;
    ConcatLayer *concat_layer;
    YOLOv3Layer *yolov3_layer;
};

// Base layer
class BaseLayer {
public:
    ~BaseLayer();
    BaseLayer();
    BaseLayer(BaseLayer *L);
    BaseLayer(const BaseLayer &L);
    BaseLayer(BaseLayer &&L);
    BaseLayer& operator=(const BaseLayer &L);
    void shape();
    string type_to_string();
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* shortcut_tensor_ = nullptr, Forward_Args *args = &default_forward_args);
    int getParameter(int type);
    void ClearGrad();
    int size() {return info.output_width * info.output_height * info.output_dimension;}
    bool save(FILE *f);
    bool load(FILE *f);
    bool load_raw(FILE *f);
    Train_Args getTrainArgs();
//protected:
    LayerType type;
    string name;
    string input_name;
    struct Info {
        int input_number;
        int output_number;
        int output_width;
        int output_height;
        int output_dimension;
        int input_width;
        int input_height;
        int input_dimension;
        int concat_dimension;
        int shortcut_width;
        int shortcut_height;
        int shortcut_dimension;
        int kernel_width;
        int kernel_height;
        int stride;
        int padding;
        int workspace_size;
        int batch_size;
        int kernel_num;
        int total_anchor_num;
        int anchor_num;
        int classes;
        int max_boxes;
        int net_width;
        int net_height;
        bool reverse;
        bool batchnorm;
    } info;
    LayerOption opt;
    Tensor* input_tensor;
    Tensor* output_tensor;
    Tensor* kernel;
    Tensor* biases;
};

// Input layer
class InputLayer : public BaseLayer {
public:
    InputLayer(LayerOption opt_);
    void Forward(Tensor *input_tensor_);
    void Backward() {}
};

// Convolution layer
class ConvolutionLayer : public BaseLayer {
public:
    ConvolutionLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* shortcut_tensor_, Forward_Args *args);
    void Forward();
    void Backward();
private:
    float *workspace;
};

// Pooling layer
class PoolingLayer : public BaseLayer {
public:
    PoolingLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

// FullyConnected layer
class FullyConnectedLayer : public BaseLayer {
public:
    FullyConnectedLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

// Relu layer
class ReluLayer : public BaseLayer {
public:
    ReluLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

// PRelu layer
class PReluLayer : public BaseLayer {
public:
    PReluLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

// Softmax layer with cross entropy loss
class SoftmaxLayer : public BaseLayer {
public:
    SoftmaxLayer(LayerOption opt_);
    void Forward();
    float Backward(vfloat& target);
private:
};

// Euclidean loss layer
class EuclideanLossLayer : public BaseLayer {
public:
    EuclideanLossLayer(LayerOption opt_);
    void Forward();
    float Backward(vfloat& target);
private:
};

// ShortCut layer
class ShortCutLayer : public BaseLayer {
public:
    ShortCutLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* shortcut_tensor_, Forward_Args *args);
    void Forward();
    void Backward();
private:
    void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
    Tensor *shortcut_tensor;
};

// LRelu layer
class LReluLayer : public BaseLayer {
public:
    LReluLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

// Sigmoid layer
class SigmoidLayer : public BaseLayer {
public:
    SigmoidLayer(LayerOption opt_);
    void Forward();
    void Backward();
};

class BatchNormalizationlayer : public BaseLayer {
public:
    BatchNormalizationlayer(LayerOption opt_);
    void Forward(Forward_Args *args = &default_forward_args);
    void Backward();
private:
    void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
    void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
    void variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
    void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
};

class UpSampleLayer : public BaseLayer {
public:
    UpSampleLayer(LayerOption opt_);
    void Forward();
    void Backward();
private:
    void upsample(float *in, int w, int h, int c, int batch, int stride, bool forward, float scale, float *out);
    void downsample(float *src, float *dst, int batch_size, int width, int height, int dimension, int stride, bool forward);
};

class ConcatLayer : public BaseLayer {
public:
    ConcatLayer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* concat_tensor_, Forward_Args *args);
    void Forward();
    void Backward();
private:
    Tensor *concat_tensor;
};

class YOLOv3Layer : public BaseLayer {
public:
    YOLOv3Layer(LayerOption opt_);
    Tensor* connectGraph(Tensor* input_tensor_, Tensor* concat_tensor_, Forward_Args *args);
    void Forward(Forward_Args *args = &default_forward_args);
    float Backward(vfloat& target);
private:
    int entry_index(int batch, int location, int entry);
    vector<Detection> yolo_get_detection_without_correction();
    Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
    float delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride);
    void delta_yolo_class(float *output, float *delta, int index, int cls, int classes, int stride, float *avg_cat);
    int int_index(float *a, int val, int n);
    float mag_array(float *a, int n);
    
    Tensor detection;
};

void scale_bias(float *output, float *scales, int batch, int n, int size);
void add_bias(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#endif /* Layer_hpp */
