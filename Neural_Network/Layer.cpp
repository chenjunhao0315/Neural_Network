//
//  Layer.cpp
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
    type = opt_["type"];
    if (type == "Input") {
        layer.input_layer = new InputLayer(opt_);
    } else if (type == "Fullyconnected") {
        layer.fullyconnected_layer = new FullyConnectedLayer(opt_);
    } else if (type == "Relu") {
        layer.relu_layer = new ReluLayer(opt_);
    } else if (type == "Softmax") {
        layer.softmax_layer = new SoftmaxLayer(opt_);
    } else if (type == "Convolution") {
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
    } else if (type == "Convolution") {
        return layer.convolution_layer->Forward(input_tensor_);
    }
    return 0;
}

float Model_Layer::Backward(float target) {
    if (type == "Softmax") {
        return layer.softmax_layer->Backward(target);
    }
    return 0;
}

void Model_Layer::Backward() {
    if (type == "Input") {
        return layer.input_layer->Backward();
    } else if (type == "Fullyconnected") {
        return layer.fullyconnected_layer->Backward();
    } else if (type == "Relu") {
        return layer.relu_layer->Backward();
    } else if (type == "Convolution") {
        return layer.convolution_layer->Backward();
    }
}

void Model_Layer::UpdateWeight(string method, float learning_rate) {
    if (type == "Fullyconnected") {
        layer.fullyconnected_layer->UpdateWeight(method, learning_rate);
    } else if (type == "Convolution") {
        layer.convolution_layer->UpdateWeight(method, learning_rate);
    }
}

void Model_Layer::shape() {
    
    if (type == "Input") {
        return layer.input_layer->shape();
    } else if (type == "Fullyconnected") {
        return layer.fullyconnected_layer->shape();
    } else if (type == "Relu") {
        return layer.relu_layer->shape();
    } else if (type == "Softmax") {
        return layer.softmax_layer->shape();
    } else if (type == "Convolution") {
        return layer.convolution_layer->shape();
    }
    
}

int Model_Layer::getParameter(int type_) {
    if (type == "Input") {
        return layer.input_layer->getParameter(type_);
    } else if (type == "Fullyconnected") {
        return layer.fullyconnected_layer->getParameter(type_);
    } else if (type == "Relu") {
        return layer.relu_layer->getParameter(type_);
    } else if (type == "Softmax") {
        return layer.softmax_layer->getParameter(type_);
    } else if (type == "Convolution") {
        return layer.convolution_layer->getParameter(type_);
    }
    return 0;
}

void BaseLayer::shape() {
    
    printf("Type: %s\n", type.c_str());
    printf("Shape: out(%d * %d * %d)\n", info.output_width, info.output_height, info.output_dimension);
    for (int i = 0; i < kernel.size(); ++i) {
        printf("Weight:\n%d: ", i);
        kernel[i].showWeight();
    }
    printf("Bias:\n");
    biases.showWeight();
    
}

int BaseLayer::getParameter(int type) {
    switch (type) {
        case 0: return info.output_width; break;
        case 1: return info.output_height; break;
        case 2: return info.output_dimension; break;
        default: break;
    }
    return 0;
}

void BaseLayer::UpdateWeight(string method, float learning_rate) {
    if (method == "SVG") {
        vfloat &bias_weight = biases.getWeight();
        vfloat &bias_grad = biases.getDeltaWeight();
        int output_dimension = info.output_dimension;
        for (int i = 0; i < output_dimension; ++i) {
            Tensor &act_tensor = kernel[i];
            int length = (int)act_tensor.getWeight().size();
            vfloat &weight = act_tensor.getWeight();
            vfloat &grad = act_tensor.getDeltaWeight();
            for (int j = 0; j < length; j++) {
                weight[j] -= learning_rate * grad[j];
            }
            float a = bias_weight[i] -= learning_rate * bias_grad[i];
            act_tensor.clearDeltaWeight();
        }
        biases.clearDeltaWeight();
    }
}

InputLayer::InputLayer(LayerOption opt_) {
    opt = opt_;
    type = "Input";
    info.output_dimension = (opt.find("input_dimension") == opt.end()) ? atoi(opt["output_dimension"].c_str()) : atoi(opt["input_dimension"].c_str());
    info.output_width = (opt.find("input_width") == opt.end()) ? atoi(opt["output_width"].c_str()) : atoi(opt["input_width"].c_str());
    info.output_height = (opt.find("input_height") == opt.end()) ? atoi(opt["output_height"].c_str()) : atoi(opt["input_height"].c_str());
}

Tensor* InputLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    output_tensor = input_tensor_;
    return output_tensor;
}

ConvolutionLayer::ConvolutionLayer(LayerOption opt_) {
    opt = opt_;
    type = "Convolution";
    
    info.output_dimension = atoi(opt["number_kernel"].c_str());
    kernel_width = atoi(opt["kernel_width"].c_str());
    input_dimension = atoi(opt["input_dimension"].c_str());
    input_width = atoi(opt["input_width"].c_str());;
    input_height = atoi(opt["input_height"].c_str());
    
    kernel_height = (opt.find("kernel_height") == opt.end()) ? kernel_width : atoi(opt["kernel_height"].c_str());
    stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    padding = (opt.find("padding") == opt.end()) ? 0 : atoi(opt["padding"].c_str());
    
    info.output_width = (input_width + padding * 2 - kernel_width) / stride + 1;
    info.output_height = (input_height + padding * 2 - kernel_height) / stride + 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    kernel.clear();
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(kernel_width, kernel_height, input_dimension);
        kernel.push_back(new_kernel);
    }
    biases = Tensor(1, 1, info.output_dimension, bias);
}

Tensor* ConvolutionLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *act_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0);
    
    int input_width = input_tensor->getWidth();
    int input_height = input_tensor->getHeight();
    int output_width = info.output_width;
    int output_height = info.output_height;
    int xy_stride = stride;
    vfloat &input_weight = input_tensor->getWeight();
    
    for (int output_d = 0; output_d < info.output_dimension; ++output_d) {
        Tensor &act_kernel = kernel[output_d];
        vfloat &bias = biases.getWeight();
        int offset_w = -padding;
        int offset_h = -padding;
        int kernel_width = act_kernel.getWidth();
        int kernel_height = act_kernel.getHeight();
        int kernel_dimension = act_kernel.getDimension();
        vfloat &kernel_weight = act_kernel.getWeight();
        for (int output_h = 0; output_h < output_height; ++output_h, offset_h += xy_stride) {
            offset_w = -padding;
            for (int output_w = 0; output_w < output_width; ++output_w, offset_w += xy_stride) {
                float sum = 0.0;
                for (int kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    int act_h = kernel_h + offset_h;
                    for (int kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        int act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < input_height && act_w >= 0 && act_w < input_width) {
                            for (int kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
                                sum += kernel_weight[((kernel_width * kernel_h) + kernel_w) * kernel_dimension + kernel_d] * input_weight[((input_width * act_h) + act_w) * input_dimension + kernel_d];
                            }
                            
                        }
                    }
                }
                sum += bias[output_d];
                act_tensor->set(output_w, output_h, output_d, sum);
            }
        }
    }
    output_tensor = act_tensor;
    return output_tensor;
}

void ConvolutionLayer::Backward() {
    input_tensor->clearDeltaWeight();
    int input_width = input_tensor->getWidth();
    int input_height = input_tensor->getHeight();
    int xy_stride = stride;
    vfloat &input_weight = input_tensor->getWeight();
    vfloat &input_grad = input_tensor->getDeltaWeight();
    
    for (int input_d = 0; input_d < input_dimension; ++input_d) {
        Tensor &act_kernel = kernel[input_d];
        vfloat &bias_grad = biases.getDeltaWeight();
        int offset_w = -padding;
        int offset_h = -padding;
        int kernel_width = act_kernel.getWidth();
        int kernel_height = act_kernel.getHeight();
        int kernel_dimension = act_kernel.getDimension();
        vfloat &kernel_weight = act_kernel.getWeight();
        vfloat &kernel_grad = act_kernel.getDeltaWeight();
        for (int input_h = 0; input_h < input_height; ++input_h, offset_h += xy_stride) {
            offset_w = -padding;
            for (int input_w = 0; input_w < input_width; ++input_w, offset_w += xy_stride) {
                float chain_grad = output_tensor->getGrad(input_w, input_h, input_d);
                for (int kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    int act_h = kernel_h + offset_h;
                    for (int kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        int act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < input_height && act_w >= 0 && act_w < input_width) {
                            for (int kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
                                int index_1 = ((input_width * input_h) + input_w) * input_dimension + kernel_d;
                                int index_2 = ((kernel_width * kernel_h) + kernel_w) * kernel_dimension + kernel_d;
                                kernel_grad[index_2] += input_weight[index_1] * chain_grad;
                                input_grad[index_1] += kernel_weight[index_2] * chain_grad;
                            }
                            
                        }
                    }
                }
                bias_grad[input_d] += chain_grad;
            }
        }
    }
}

FullyConnectedLayer::FullyConnectedLayer(LayerOption opt_) {
    opt = opt_;
    type = "Fullyconnected";
    info.output_dimension = atoi(opt["number_neurons"].c_str());
    
    int a;
    a = info.input_number = atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str()) * atoi(opt["input_dimension"].c_str());
    info.output_width = 1;
    info.output_height = 1;
    
    kernel.clear();
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(1, 1, info.input_number);
        kernel.push_back(new_kernel);
    }
    float bias = atof(opt["bias"].c_str());
    biases = new Tensor(1, 1, info.output_dimension, bias);
}

Tensor* FullyConnectedLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(1, 1, info.output_dimension, 0);
    vfloat &pos = cal->getWeight();
    vfloat &input = input_tensor_->getWeight();
    vfloat &bias = biases.getWeight();
    int output_dimension = info.output_dimension;
    int input_number = info.input_number;
    
    // get output
    for (int i = 0; i < output_dimension; ++i) {
        float sum = 0.0;
        vfloat &weight = kernel[i].getWeight();
        for (int j = 0; j < input_number; ++j) {
            sum += input[j] * weight[j];
        }
        
        sum += bias[i];
        pos[i] = sum;
    }
    output_tensor = cal;
    return output_tensor;
}

void FullyConnectedLayer::Backward() {
    Tensor *cal = input_tensor;
    cal->clearDeltaWeight();
    vfloat &cal_w = cal->getWeight();
    vfloat &cal_dw = cal->getDeltaWeight();
    vfloat &act_biases_grad = biases.getDeltaWeight();
    int output_dimension = info.output_dimension;
    vfloat &output_grad = output_tensor->getDeltaWeight();
    int input_number = info.input_number;
    
    for (int i = 0; i < output_dimension; ++i) {
        vfloat &act_weight = kernel[i].getWeight();
        vfloat &act_grad = kernel[i].getDeltaWeight();
        float chain_grad = output_grad[i];
        for (int j = 0; j < input_number; ++j) {
            cal_dw[j] += act_weight[j] * chain_grad;
            act_grad[j] += cal_w[j] * chain_grad;
        }
        act_biases_grad[i] += chain_grad;
    }
}

ReluLayer::ReluLayer(LayerOption opt_) {
    opt = opt_;
    type = "Relu";
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
}

Tensor* ReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(input_tensor_);
    int length = (int)cal->getWeight().size();
    vfloat &val = cal->getWeight();
    for (int i = 0; i < length; ++i) {
        if (val[i] < 0)
            val[i] = 0;
    }
    output_tensor = cal;
    return output_tensor;
}

void ReluLayer::Backward() {
    Tensor *cal = input_tensor;
    cal->clearDeltaWeight();
    int length = (int)cal->getWeight().size();
    vfloat &act_weight = output_tensor->getWeight();
    vfloat &act_grad = output_tensor->getDeltaWeight();
    vfloat &pos_grad = cal->getDeltaWeight();
    
    for (int i = 0; i < length; ++i) {
        if (act_weight[i] <= 0)
            pos_grad[i] = 0;
        else
            pos_grad[i] = act_grad[i];
    }
}

SoftmaxLayer::SoftmaxLayer(LayerOption opt_) {
    opt = opt_;
    type = "Softmax";
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str()) * info.output_dimension;
    info.output_width = 1;
    info.output_height = 1;
}

Tensor* SoftmaxLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(1, 1, info.output_dimension, 0);
    int output_dimension = info.output_dimension;
    
    vfloat &act = input_tensor_->getWeight();
    float max = act[0];
    for (int i = 1; i < output_dimension; ++i) {
        if (act[i] > max)
            max = act[i];
    }
    vfloat expo_sum_(output_dimension, 0);
    float sum = 0;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indiviual = exp(act[i] - max);
        sum += indiviual;
        expo_sum_[i] = indiviual;
    }
    
    vfloat &cal_weight = cal->getWeight();
    for (int i = 0; i < output_dimension; ++i) {
        expo_sum_[i] /= sum;
        cal_weight[i] = expo_sum_[i];
    }
    
    expo_sum = expo_sum_;
    output_tensor = cal;
    
    return output_tensor;
}

float SoftmaxLayer::Backward(float target) {
    Tensor *cal_tensor = input_tensor;
    cal_tensor->clearDeltaWeight();
    vfloat &cal_delta_weight = cal_tensor->getDeltaWeight();
    int output_dimension = info.output_dimension;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indicator = (i == target) ? 1.0 : 0.0;
        float mul = -(indicator - expo_sum[i]);
        cal_delta_weight[i] = mul;
    }
    return -log(expo_sum[(int)target]);
}
