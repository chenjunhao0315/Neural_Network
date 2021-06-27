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

#include "Tensor.hpp"

using namespace std;

typedef vector<Tensor> vtensor;
typedef map<string, string> LayerOption;

class InputLayer;
class FullyConnectedLayer;
class ReluLayer;
class SoftmaxLayer;
class ConvolutionLayer;
class PoolingLayer;

// Top layer
class Model_Layer {
public:
    ~Model_Layer();
    Model_Layer();
    Model_Layer(const Model_Layer &L);
    Model_Layer(Model_Layer &&L);
    Model_Layer& operator=(const Model_Layer &L);
    Model_Layer(LayerOption opt_);
    Tensor* Forward(Tensor* input_tensor_);
    float Backward(float target);
    void Backward();
    void UpdateWeight(string method, float learning_rate);
    void shape();
    int getParameter(int type_);
    string getType() {return type;}
    bool save(FILE *f);
    bool load(FILE *f);
private:
    string type;
    InputLayer *input_layer;
    FullyConnectedLayer *fullyconnected_layer;
    ReluLayer *relu_layer;
    SoftmaxLayer *softmax_layer;
    ConvolutionLayer *convolution_layer;
    PoolingLayer *pooling_layer;
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
    int getParameter(int type);
    void UpdateWeight(string method, float learning_rate);
    int size() {return info.output_width * info.output_height * info.output_dimension;}
    bool save(FILE *f);
    bool load(FILE *f);
protected:
    string type;
    struct info_ {
        int input_number;
        int output_width;
        int output_height;
        int output_dimension;
    } info;
    LayerOption opt;
    Tensor* input_tensor;
    Tensor* output_tensor;
    Tensor* kernel;
    Tensor biases;
    struct info_more {
        int input_width;
        int input_height;
        int input_dimension;
        int kernel_width;
        int kernel_height;
        int stride;
        int padding;
    } info_more;
};

// Input layer
class InputLayer : public BaseLayer {
public:
    ~InputLayer() {output_tensor = nullptr;}
    InputLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    void Backward() {}
};

// Convolution layer
class ConvolutionLayer : public BaseLayer {
public:
    ConvolutionLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    void Backward();
private:
};

class PoolingLayer : public BaseLayer {
public:
    PoolingLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    void Backward();
private:
    vector<int> choosex;
    vector<int> choosey;
};

// FullyConnected layer
class FullyConnectedLayer : public BaseLayer {
public:
    FullyConnectedLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    void Backward();
};

// Relu layer
class ReluLayer : public BaseLayer {
public:
    ReluLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    void Backward();
};

// Softmax layer
class SoftmaxLayer : public BaseLayer {
public:
    ~SoftmaxLayer() {delete [] expo_sum; expo_sum = nullptr;}
    SoftmaxLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    float Backward(float target);
private:
    float* expo_sum;
};

#endif /* Layer_hpp */
