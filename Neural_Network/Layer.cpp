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

//Model_Layer::~Model_Layer() {
//    if (type == "Input") {
//        delete layer.input_layer;
//    } else if (type == "Fullyconnected") {
//        delete layer.fullyconnected_layer;
//    } else if (type == "Relu") {
//        delete layer.relu_layer;
//    } else if (type == "Softmax") {
//        delete layer.softmax_layer;
//    } else if (type == "Convolution") {
//        delete layer.convolution_layer;
//    } else if (type == "Pooling") {
//        delete layer.pooling_layer;
//    }
//}

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
    } else if (type == "Pooling") {
        layer.pooling_layer = new PoolingLayer(opt_);
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
    } else if (type == "Pooling") {
        return layer.pooling_layer->Forward(input_tensor_);
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
    } else if (type == "Pooling") {
        layer.pooling_layer->Backward();
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
    } else if (type == "Pooling") {
        return layer.pooling_layer->shape();
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
    } else if (type == "Pooling") {
        return layer.pooling_layer->getParameter(type_);
    }
    return 0;
}

void BaseLayer::shape() {
    
    printf("Type: %s\n", type.c_str());
    printf("Shape: out(%d * %d * %d)\n", info.output_width, info.output_height, info.output_dimension);
//    if (type == "Convolution" || type == "Fullyconnected") {
//        for (int i = 0; i < info.output_dimension; ++i) {
//            printf("Weight:\n%d: ", i);
//            kernel[i].showWeight();
//        }
//        printf("Bias:\n");
//        biases.showWeight();
//    }
    
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
        float *bias_weight = biases.getWeight();
        float *bias_grad = biases.getDeltaWeight();
        int output_dimension = info.output_dimension;
        for (int i = 0; i < output_dimension; ++i) {
            Tensor *act_tensor = kernel + i;
            int length = (int)act_tensor->length();
            float *weight = act_tensor->getWeight();
            float *grad = act_tensor->getDeltaWeight();
            for (int j = 0; j < length; j++) {
                weight[j] -= learning_rate * grad[j];
            }
            bias_weight[i] -= learning_rate * bias_grad[i];
            act_tensor->clearDeltaWeight();
        }
        biases.clearDeltaWeight();
    }
}

void BaseLayer::save() {
    fstream model("model.bin", ios::out | ios::app | ios::binary);
    if (type == "Fullyconnected") {
        model.write("fc", sizeof(char) * 2);
        model.write((char *)&info.output_width, sizeof(int));
        model.write((char *)&info.output_height, sizeof(int));
        model.write((char *)&info.output_dimension, sizeof(int));
    }
    model.close();
}

bool BaseLayer::load() {
    fstream model("model.bin", ios::in | ios::binary);
    return true;
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
    input_width = atoi(opt["input_width"].c_str());
    input_height = atoi(opt["input_height"].c_str());
    
    kernel_height = (opt.find("kernel_height") == opt.end()) ? kernel_width : atoi(opt["kernel_height"].c_str());
    stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    padding = (opt.find("padding") == opt.end()) ? 0 : atoi(opt["padding"].c_str());
    
    info.output_width = (input_width + padding * 2 - kernel_width) / stride + 1;
    info.output_height = (input_height + padding * 2 - kernel_height) / stride + 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(kernel_width, kernel_height, input_dimension);
        kernel[i] = new_kernel;
    }
    Tensor new_bias(1, 1, info.output_dimension, bias);
    biases = new_bias;
}

Tensor* ConvolutionLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *act_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0);
    
    int input_width = input_tensor->getWidth();
    int input_height = input_tensor->getHeight();
    int output_width = info.output_width;
    int output_height = info.output_height;
    int xy_stride = stride;
    float *input_weight = input_tensor->getWeight();
    
    int output_d, output_w, output_h;
    int kernel_d, kernel_w, kernel_h;
    int offset_w, offset_h;
    int kernel_width, kernel_height, kernel_dimension;
    float sum;
    int act_h, act_w;
    float *bias;
    float *kernel_weight;
    
    for (output_d = 0; output_d < info.output_dimension; ++output_d) {
        Tensor *act_kernel = kernel + output_d;
        bias = biases.getWeight();
        offset_h = -padding;
        kernel_width = act_kernel->getWidth();
        kernel_height = act_kernel->getHeight();
        kernel_dimension = act_kernel->getDimension();
        kernel_weight = act_kernel->getWeight();
        for (output_h = 0; output_h < output_height; ++output_h, offset_h += xy_stride) {
            offset_w = -padding;
            for (output_w = 0; output_w < output_width; ++output_w, offset_w += xy_stride) {
                sum = 0.0;
                for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    act_h = kernel_h + offset_h;
                    for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < input_height && act_w >= 0 && act_w < input_width) {
                            for (kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
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
    if (output_tensor)
        delete output_tensor;
    output_tensor = act_tensor;
    return output_tensor;
}

void ConvolutionLayer::Backward() {
    input_tensor->clearDeltaWeight();
//    int input_width = input_tensor->getWidth();
//    int input_height = input_tensor->getHeight();
    int xy_stride = stride;
    float *input_weight = input_tensor->getWeight();
    float *input_grad = input_tensor->getDeltaWeight();
    
    int output_width = output_tensor->getWidth();
    int output_height = output_tensor->getHeight();
    int output_d, output_w, output_h;
    int kernel_d, kernel_w, kernel_h;
    int offset_w, offset_h;
    int kernel_width, kernel_height, kernel_dimension;
    int act_h, act_w;
    float *bias_grad;
    float *kernel_weight, *kernel_grad;
    Tensor *act_kernel;
    int index_1, index_2;
    
    for (output_d = 0; output_d < info.output_dimension; ++output_d) {
        act_kernel = kernel + output_d;
        bias_grad = biases.getDeltaWeight();
        offset_h = -padding;
        kernel_width = act_kernel->getWidth();
        kernel_height = act_kernel->getHeight();
        kernel_dimension = act_kernel->getDimension();
        kernel_weight = act_kernel->getWeight();
        kernel_grad = act_kernel->getDeltaWeight();
        for (output_h = 0; output_h < output_height; ++output_h, offset_h += xy_stride) {
            offset_w = -padding;
            for (output_w = 0; output_w < output_width; ++output_w, offset_w += xy_stride) {
                float chain_grad = output_tensor->getGrad(output_w, output_h, output_d);
                for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    act_h = kernel_h + offset_h;
                    for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < input_height && act_w >= 0 && act_w < input_width) {
                            for (kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
                                index_1 = ((input_width * output_h) + output_w) * input_dimension + kernel_d;
                                index_2 = ((kernel_width * kernel_h) + kernel_w) * kernel_dimension + kernel_d;
                                kernel_grad[index_2] += input_weight[index_1] * chain_grad;
                                input_grad[index_1] += kernel_weight[index_2] * chain_grad;
                            }
                            
                        }
                    }
                }
                bias_grad[output_d] += chain_grad;
            }
        }
    }
}

PoolingLayer::PoolingLayer(LayerOption opt_) {
    opt = opt_;
    type = "Pooling";
    
    kernel_width = atoi(opt["kernel_width"].c_str());
    input_dimension = atoi(opt["input_dimension"].c_str());
    input_width = atoi(opt["input_width"].c_str());
    input_height = atoi(opt["input_height"].c_str());
    
    kernel_height = (opt.find("kernel_height") == opt.end()) ? kernel_width : atoi(opt["kernel_height"].c_str());
    stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    padding = (opt.find("padding") == opt.end()) ? 0 : atoi(opt["padding"].c_str());
    
    info.output_dimension = input_dimension;
    info.output_width = (input_width + padding * 2 - kernel_width) / stride + 1;
    info.output_height = (input_height + padding * 2 - kernel_height) / stride + 1;
    
    choosex.assign(info.output_width * info.output_height * info.output_dimension, 0);
    choosey.assign(info.output_width * info.output_height * info.output_dimension, 0);
}

Tensor* PoolingLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
//    printf("input:\n");
//    input_tensor->showWeight();
    Tensor *act_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0.0);
    int output_dimension = info.output_dimension;
    int output_width = info.output_width;
    int output_height = info.output_height;
    
    int output_d, output_w, output_h;
    int offset_w, offset_h;
    float minimum;
    int win_x, win_y;
    int kernel_w, kernel_h;
    int act_w, act_h;
    float value = 0.0;
    
    int counter = 0;
    for (output_d = 0; output_d < output_dimension; ++output_d) {
        offset_w = -padding;
        for (output_w = 0; output_w < output_width; ++output_w, offset_w += stride) {
            offset_h = -padding;
            for (output_h = 0; output_h < output_height; ++output_h, offset_h += stride) {
                minimum = -100000000;
                win_x = -1;
                win_y = -1;
                for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                    for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                        act_w = offset_w + output_w;
                        act_h = offset_h + output_h;
                        if (act_w >= 0 && act_w < input_width && act_h >= 0 && act_h < input_height) {
                            value = input_tensor->get(act_w, act_h, output_d);
                            if (value > minimum) {
                                minimum = value;
                                win_x = act_w;
                                win_y = act_h;
                            }
                        }
                    }
                }
                choosex[counter] = win_x;
                choosey[counter] = win_y;
                ++counter;
//                printf("value: %f\n", value);
                act_tensor->set(output_w, output_h, output_d, value);
            }
        }
    }
    if (output_tensor)
        delete output_tensor;
    output_tensor = act_tensor;
    return output_tensor;
}

void PoolingLayer::Backward() {
    Tensor *input = input_tensor;
    input->clearDeltaWeight();
    Tensor *output = output_tensor;
    
    int output_dimension = info.output_dimension;
    int output_width = info.output_width;
    int output_height = info.output_height;
    int counter = 0;
    int cal_d, cal_w, cal_h;
    int offset_w, offset_h;
    float chain_grad;
    
    for (cal_d = 0; cal_d < output_dimension; ++cal_d) {
        offset_w = -padding;
        for (cal_w = 0; cal_w < output_width; ++cal_w, offset_w += stride) {
            offset_h = -padding;
            for (cal_h = 0; cal_h < output_height; ++cal_h, offset_h += stride) {
                chain_grad = output->getGrad(cal_w, cal_h, cal_d);
                input->addGrad(choosex[counter], choosey[counter], cal_d, chain_grad);
                counter++;
            }
        }
    }
}

FullyConnectedLayer::FullyConnectedLayer(LayerOption opt_) {
    opt = opt_;
    type = "Fullyconnected";
    info.output_dimension = atoi(opt["number_neurons"].c_str());
    
    info.input_number = atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str()) * atoi(opt["input_dimension"].c_str());
    info.output_width = 1;
    info.output_height = 1;
    
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(1, 1, info.input_number);
        kernel[i] = new_kernel;
    }
    float bias = atof(opt["bias"].c_str());
    Tensor new_bias(1, 1, info.output_dimension, bias);
    biases = new_bias;
}

Tensor* FullyConnectedLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(1, 1, info.output_dimension, 0);
    float *pos = cal->getWeight();
    float *input = input_tensor_->getWeight();
    float *bias = biases.getWeight();
    int output_dimension = info.output_dimension;
    int input_number = info.input_number;
    
    float sum;
    float *weight;
    
    // get output
    for (int i = 0; i < output_dimension; ++i) {
        sum = 0.0;
        weight = kernel[i].getWeight();
        for (int j = 0; j < input_number; ++j) {
            sum += input[j] * weight[j];
        }
        
        sum += bias[i];
        pos[i] = sum;
    }
    if (output_tensor)
        delete output_tensor;
    output_tensor = cal;
    return output_tensor;
}

void FullyConnectedLayer::Backward() {
    Tensor *cal = input_tensor;
    cal->clearDeltaWeight();
    float *cal_w = cal->getWeight();
    float *cal_dw = cal->getDeltaWeight();
    float *act_biases_grad = biases.getDeltaWeight();
    int output_dimension = info.output_dimension;
    float *output_grad = output_tensor->getDeltaWeight();
    int input_number = info.input_number;
    
    float *act_weight, *act_grad;
    float chain_grad;
    
    for (int i = 0; i < output_dimension; ++i) {
        act_weight = kernel[i].getWeight();
        act_grad = kernel[i].getDeltaWeight();
        chain_grad = output_grad[i];
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
    int length = (int)cal->length();
    float *val = cal->getWeight();
    for (int i = 0; i < length; ++i) {
        if (val[i] < 0)
            val[i] = 0;
    }
    if (output_tensor)
        delete output_tensor;
    output_tensor = cal;
    return output_tensor;
}

void ReluLayer::Backward() {
    Tensor *cal = input_tensor;
    cal->clearDeltaWeight();
    int length = (int)cal->length();
    float *act_weight = output_tensor->getWeight();
    float *act_grad = output_tensor->getDeltaWeight();
    float *pos_grad = cal->getDeltaWeight();
    
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
    expo_sum = nullptr;
}

Tensor* SoftmaxLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(1, 1, info.output_dimension, 0);
    int output_dimension = info.output_dimension;
    
    float *act = input_tensor_->getWeight();
    float max = act[0];
    for (int i = 1; i < output_dimension; ++i) {
        if (act[i] > max)
            max = act[i];
    }
    float *expo_sum_ = new float [output_dimension];
    fill (expo_sum_, expo_sum_ + output_dimension, 0);
//    memset(expo_sum_, 0, sizeof(float) * output_dimension);
    float sum = 0;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indiviual = exp(act[i] - max);
        sum += indiviual;
        expo_sum_[i] = indiviual;
    }
    
    float *cal_weight = cal->getWeight();
    for (int i = 0; i < output_dimension; ++i) {
        expo_sum_[i] /= sum;
        cal_weight[i] = expo_sum_[i];
    }
    
    if (expo_sum)
        delete [] expo_sum;
    expo_sum = expo_sum_;
    
    if (output_tensor)
        delete output_tensor;
    output_tensor = cal;
    
    return output_tensor;
}

float SoftmaxLayer::Backward(float target) {
    Tensor *cal_tensor = input_tensor;
    cal_tensor->clearDeltaWeight();
    float *cal_delta_weight = cal_tensor->getDeltaWeight();
    int output_dimension = info.output_dimension;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indicator = (i == target) ? 1.0 : 0.0;
        float mul = -(indicator - expo_sum[i]);
        cal_delta_weight[i] = mul;
    }
    return -log(expo_sum[(int)target]);
}
