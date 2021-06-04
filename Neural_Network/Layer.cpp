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
