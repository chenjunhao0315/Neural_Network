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

#include "Tensor.hpp"

using namespace std;

typedef vector<Tensor> vtensor;
typedef map<string, string> LayerOption;

class InputLayer;
class FullyConnectedLayer;
class ReluLayer;
class SoftmaxLayer;
class ConvolutionLayer;

// Top layer
class Model_Layer {
public:
    Model_Layer() {}
    Model_Layer(LayerOption opt_);
    Tensor* Forward(Tensor* input_tensor_);
    float Backward(float target);
    void Backward();
    void UpdateWeight(string method, float learning_rate);
    void shape();
    int getParameter(int type);
    string getType() {return type;}
private:
    string type;
    union {
        InputLayer *input_layer;
        FullyConnectedLayer *fullyconnected_layer;
        ReluLayer *relu_layer;
        SoftmaxLayer *softmax_layer;
        ConvolutionLayer *convolution_layer;
    } layer;
};

// Base layer
class BaseLayer {
public:
    BaseLayer() {}
    void shape();
    int getParameter(int type);
    void UpdateWeight(string method, float learning_rate);
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
    vtensor kernel;
    Tensor biases;
};

// Input layer
class InputLayer : public BaseLayer {
public:
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
    int input_width;
    int input_height;
    int input_dimension;
    int kernel_width;
    int kernel_height;
    int stride;
    int padding;
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
    SoftmaxLayer(LayerOption opt_);
    Tensor* Forward(Tensor *input_tensor_);
    float Backward(float target);
private:
    vfloat expo_sum;
};

#endif /* Layer_hpp */
