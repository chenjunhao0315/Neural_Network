//
//  Tensor.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Layer.hpp"

class InputLayer;
class FullyConnectedLayer;
class ReluLayer;
class SoftmaxLayer;
class ConvolutionLayer;

Model_Layer::Model_Layer(LayerOption opt_) {
    string type = opt_["type"];
    if (type == "Input") {
        layer.input_layer = new InputLayer(opt_);
    } else if (type == "Fullyconnected") {
        layer.fullyconnected_layer = new FullyConnectedLayer(opt_);
    } else if (type == "Relu") {
        layer.relu_layer = new ReluLayer(opt_);
    } else if (type == "Softmax") {
        layer.softmax_layer = new SoftmaxLayer(opt_);
    } else if (type == "Convloution") {
        layer.convolution_layer = new ConvolutionLayer(opt_);
    }
}

Tensor* Model_Layer::Forward(Tensor* input_tensor_) {
    if (type == "Input") {
        return layer.input_layer->Forward(input_tensor_);
    } else if (type == "Fullyconnected") {
        return layer.fullyconnected_layer->Forward(input_tensor_);
    } else if (type == "Relu") {
        return layer.relu_layer->Forward(input_tensor_);
    } else if (type == "Softmax") {
        return layer.softmax_layer->Forward(input_tensor_);
    } else if (type == "Convloution") {
        return layer.convolution_layer->Forward(input_tensor_);
    }
    return 0;
}
