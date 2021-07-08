//
//  Layer.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Layer.hpp"

Model_Layer::~Model_Layer() {
    if (type == LayerType::Input) {
        delete input_layer;
    } else if (type == LayerType::Fullyconnected) {
        delete fullyconnected_layer;
    } else if (type == LayerType::Relu) {
        delete relu_layer;
    } else if (type == LayerType::Softmax) {
        delete softmax_layer;
    } else if (type == LayerType::Convolution) {
        delete convolution_layer;
    } else if (type == LayerType::Pooling) {
        delete pooling_layer;
    } else if (type == LayerType::EuclideanLoss) {
        delete euclideanloss_layer;
    } else if (type == LayerType::PRelu) {
        delete prelu_layer;
    }
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
}

Model_Layer::Model_Layer() {
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
}

Model_Layer::Model_Layer(const Model_Layer &L) {
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
    if (this != &L) {
        type = L.type;
        if (type == LayerType::Input) {
            input_layer = new InputLayer(*L.input_layer);
        } else if (type == LayerType::Fullyconnected) {
            fullyconnected_layer = new FullyConnectedLayer(*L.fullyconnected_layer);
        } else if (type == LayerType::Relu) {
            relu_layer = new ReluLayer(*L.relu_layer);
        } else if (type == LayerType::Softmax) {
            softmax_layer = new SoftmaxLayer(*L.softmax_layer);
        } else if (type == LayerType::Convolution) {
            convolution_layer = new ConvolutionLayer(*L.convolution_layer);
        } else if (type == LayerType::Pooling) {
            pooling_layer = new PoolingLayer(*L.pooling_layer);
        } else if (type == LayerType::EuclideanLoss) {
            euclideanloss_layer = new EuclideanLossLayer(*L.euclideanloss_layer);
        } else if (type == LayerType::PRelu) {
            prelu_layer = new PReluLayer(*L.prelu_layer);
        }
    }
}

Model_Layer::Model_Layer(Model_Layer &&L) {
    type = L.type;
    input_layer = L.input_layer;
    fullyconnected_layer = L.fullyconnected_layer;
    relu_layer = L.relu_layer;
    prelu_layer = L.prelu_layer;
    softmax_layer = L.softmax_layer;
    convolution_layer = L.convolution_layer;
    pooling_layer = L.pooling_layer;
    euclideanloss_layer = L.euclideanloss_layer;
    L.input_layer = nullptr;
    L.fullyconnected_layer = nullptr;
    L.relu_layer = nullptr;
    L.prelu_layer = nullptr;
    L.softmax_layer = nullptr;
    L.convolution_layer = nullptr;
    L.pooling_layer = nullptr;
    L.euclideanloss_layer = nullptr;
}

Model_Layer& Model_Layer::operator=(const Model_Layer &L) {
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
    if (this != &L) {
        type = L.type;
        if (type == LayerType::Input) {
            input_layer = new InputLayer(*L.input_layer);
        } else if (type == LayerType::Fullyconnected) {
            fullyconnected_layer = new FullyConnectedLayer(*L.fullyconnected_layer);
        } else if (type == LayerType::Relu) {
            relu_layer = new ReluLayer(*L.relu_layer);
        } else if (type == LayerType::Softmax) {
            softmax_layer = new SoftmaxLayer(*L.softmax_layer);
        } else if (type == LayerType::Convolution) {
            convolution_layer = new ConvolutionLayer(*L.convolution_layer);
        } else if (type == LayerType::Pooling) {
            pooling_layer = new PoolingLayer(*L.pooling_layer);
        } else if (type == LayerType::EuclideanLoss) {
            euclideanloss_layer = new EuclideanLossLayer(*L.euclideanloss_layer);
        } else if (type == LayerType::PRelu) {
            prelu_layer = new PReluLayer(*L.prelu_layer);
        }
    }
    return *this;
}

Model_Layer::Model_Layer(LayerOption opt_) {
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
    type = string_to_type(opt_["type"]);
    if (type == LayerType::Input) {
        input_layer = new InputLayer(opt_);
    } else if (type == LayerType::Fullyconnected) {
        fullyconnected_layer = new FullyConnectedLayer(opt_);
    } else if (type == LayerType::Relu) {
        relu_layer = new ReluLayer(opt_);
    } else if (type == LayerType::Softmax) {
        softmax_layer = new SoftmaxLayer(opt_);
    } else if (type == LayerType::Convolution) {
        convolution_layer = new ConvolutionLayer(opt_);
    } else if (type == LayerType::Pooling) {
        pooling_layer = new PoolingLayer(opt_);
    } else if (type == LayerType::EuclideanLoss) {
        euclideanloss_layer = new EuclideanLossLayer(opt_);
    } else if (type == LayerType::PRelu) {
        prelu_layer = new PReluLayer(opt_);
    }
}

LayerType Model_Layer::string_to_type(string type) {
    if (type == "Input") {
        return LayerType::Input;
    } else if (type == "Fullyconnected") {
        return LayerType::Fullyconnected;
    } else if (type == "Relu") {
        return LayerType::Relu;
    } else if (type == "PRelu") {
        return LayerType::PRelu;
    } else if (type == "Softmax") {
        return LayerType::Softmax;
    } else if (type == "Convolution") {
        return LayerType::Convolution;
    } else if (type == "Pooling") {
        return LayerType::Pooling;
    } else if (type == "EuclideanLoss") {
        return LayerType::EuclideanLoss;
    }
    return LayerType::Error;
}

Tensor* Model_Layer::Forward(Tensor* input_tensor_) {
    if (type == LayerType::Input) {
        return input_layer->Forward(input_tensor_);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->Forward(input_tensor_);
    } else if (type == LayerType::Relu) {
        return relu_layer->Forward(input_tensor_);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->Forward(input_tensor_);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->Forward(input_tensor_);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->Forward(input_tensor_);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->Forward(input_tensor_);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->Forward(input_tensor_);
    }
    return nullptr;
}

float Model_Layer::Backward(vfloat& target) {
    if (type == LayerType::Softmax) {
        return softmax_layer->Backward(target);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->Backward(target);
    }
    return 0;
}

void Model_Layer::Backward() {
    if (type == LayerType::Input) {
        return input_layer->Backward();
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->Backward();
    } else if (type == LayerType::Relu) {
        return relu_layer->Backward();
    } else if (type == LayerType::Convolution) {
        return convolution_layer->Backward();
    } else if (type == LayerType::Pooling) {
        pooling_layer->Backward();
    } else if (type == LayerType::PRelu) {
        prelu_layer->Backward();
    }
}

void Model_Layer::UpdateWeight(string method, float learning_rate) {
    if (type == LayerType::Fullyconnected) {
        fullyconnected_layer->UpdateWeight(method, learning_rate);
    } else if (type == LayerType::Convolution) {
        convolution_layer->UpdateWeight(method, learning_rate);
    } else if (type == LayerType::PRelu) {
        prelu_layer->UpdateWeight(method, learning_rate);
    }
}

void Model_Layer::shape() {
    if (type == LayerType::Input) {
        input_layer->shape();
    } else if (type == LayerType::Fullyconnected) {
        fullyconnected_layer->shape();
    } else if (type == LayerType::Relu) {
        relu_layer->shape();
    } else if (type == LayerType::Softmax) {
        softmax_layer->shape();
    } else if (type == LayerType::Convolution) {
        convolution_layer->shape();
    } else if (type == LayerType::Pooling) {
        pooling_layer->shape();
    } else if (type == LayerType::EuclideanLoss) {
        euclideanloss_layer->shape();
    } else if (type == LayerType::PRelu) {
        prelu_layer->shape();
    }
}

int Model_Layer::getParameter(int type_) {
    if (type == LayerType::Input) {
        return input_layer->getParameter(type_);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->getParameter(type_);
    } else if (type == LayerType::Relu) {
        return relu_layer->getParameter(type_);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->getParameter(type_);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->getParameter(type_);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->getParameter(type_);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->getParameter(type_);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->getParameter(type_);
    }
    return 0;
}

bool Model_Layer::save(FILE *f) {
    if (type == LayerType::Input) {
        return input_layer->save(f);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->save(f);
    } else if (type == LayerType::Relu) {
        return relu_layer->save(f);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->save(f);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->save(f);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->save(f);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->save(f);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->save(f);
    }
    return 0;
}

bool Model_Layer::load(FILE *f) {
    if (type == LayerType::Input) {
        return input_layer->load(f);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->load(f);
    } else if (type == LayerType::Relu) {
        return relu_layer->load(f);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->load(f);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->load(f);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->load(f);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->load(f);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->load(f);
    }
    return 0;
}

Tensor* Model_Layer::getKernel() {
    if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->getKernel();
    } else if (type == LayerType::Convolution) {
        return convolution_layer->getKernel();
    } else if (type == LayerType::PRelu) {
        return prelu_layer->getKernel();
    }
    return nullptr;
}

Tensor* Model_Layer::getBiases() {
    if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->getBiases();
    } else if (type == LayerType::Convolution) {
        return convolution_layer->getBiases();
    } else if (type == LayerType::PRelu) {
        return prelu_layer->getKernel();
    }
    return nullptr;
}

vfloat Model_Layer::getDetailParameter() {
    if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->getDetailParameter();
    } else if (type == LayerType::Convolution) {
        return convolution_layer->getDetailParameter();
    } else if (type == LayerType::PRelu) {
        return prelu_layer->getDetailParameter();
    }
    return vfloat();
}

Tensor* BaseLayer::getKernel() {
    return kernel;
}

Tensor* BaseLayer::getBiases() {
    return biases;
}

vfloat BaseLayer::getDetailParameter() {
    vfloat detail;
    if (type == LayerType::Fullyconnected) {
        detail.push_back(info.output_dimension);
        detail.push_back(1);
        detail.push_back(1);
        detail.push_back(info.input_number);
        detail.push_back(info.output_dimension);
        detail.push_back(0);    // l1 decay mul
        detail.push_back(1);    // l2 decay mul
    } else if (type == LayerType::Convolution) {
        detail.push_back(info.output_dimension);
        detail.push_back(info_more.kernel_width);
        detail.push_back(info_more.kernel_height);
        detail.push_back(info_more.input_dimension);
        detail.push_back(info.output_dimension);
        detail.push_back(0);    // l1 decay mul
        detail.push_back(1);    // l2 decay mul
    } else if (type == LayerType::PRelu) {
        detail.push_back(1);
        detail.push_back(1);
        detail.push_back(1);
        detail.push_back(info.input_number);
        detail.push_back(0);
        detail.push_back(0);    // l1 decay mul
        detail.push_back(0);    // l2 decay mul
    }
    return detail;
}

BaseLayer::~BaseLayer() {
    if (output_tensor)
        delete output_tensor;
    if (kernel)
        delete [] kernel;
    if (biases)
        delete biases;
}

BaseLayer::BaseLayer() {
    input_tensor = nullptr;
    output_tensor = nullptr;
    kernel = nullptr;
    biases = nullptr;
}

BaseLayer::BaseLayer(BaseLayer *L) {
    if (this != L) {
        type = L->type;
        name = L->name;
        input_name = L->input_name;
        info = L->info;
        info_more = L->info_more;
        opt = L->opt;
        input_tensor = new Tensor(L->input_tensor);
        output_tensor = new Tensor(L->output_tensor);
        if (L->kernel) {
            int len;
            if (type == LayerType::PRelu)
                len = 1;
            else
                len = info.output_dimension;
            kernel = new Tensor [len];
            for (int i = 0; i < len; ++i) {
                kernel[i] = L->kernel[i];
            }
        } else {
            kernel = nullptr;
        }
        if (L->biases) {
            biases = new Tensor(L->biases);
        } else {
            biases = nullptr;
        }
    }
}

BaseLayer::BaseLayer(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        info_more = L.info_more;
        opt = L.opt;
        input_tensor = L.input_tensor;
        output_tensor = L.output_tensor;
        if (L.kernel) {
            int len;
            if (type == LayerType::PRelu)
                len = 1;
            else
                len = info.output_dimension;
            kernel = new Tensor [len];
            for (int i = 0; i < len; ++i) {
                kernel[i] = L.kernel[i];
            }
        } else {
            kernel = nullptr;
        }
        if (L.biases) {
            biases = new Tensor(L.biases);
        } else {
            biases = nullptr;
        }
    }
}

BaseLayer::BaseLayer(BaseLayer &&L) {
    type = L.type;
    name = L.name;
    input_name = L.input_name;
    info = L.info;
    info_more = L.info_more;
    opt = L.opt;
    input_tensor = L.input_tensor;
    L.input_tensor = nullptr;
    output_tensor = L.output_tensor;
    L.output_tensor = nullptr;
    kernel = L.kernel;
    L.kernel = nullptr;
    biases = L.biases;
    L.biases = nullptr;
}

BaseLayer& BaseLayer::operator=(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        info_more = L.info_more;
        opt = L.opt;
        input_tensor = L.input_tensor;
        output_tensor = L.output_tensor;
        if (L.kernel) {
            int len;
            if (type == LayerType::PRelu)
                len = 1;
            else
                len = info.output_dimension;
            kernel = new Tensor [len];
            for (int i = 0; i < len; ++i) {
                kernel[i] = L.kernel[i];
            }
        } else {
            kernel = nullptr;
        }
        if (L.biases) {
            biases = new Tensor(L.biases);
        } else {
            biases = nullptr;
        }
    }
    return *this;
}

string BaseLayer::type_to_string() {
    switch(type) {
        case Input: return "Input"; break;
        case Fullyconnected: return "Fullyconnected"; break;
        case Relu: return "Relu"; break;
        case PRelu: return "PRelu"; break;
        case Softmax: return "Softmax"; break;
        case Convolution: return "Convolution"; break;
        case Pooling: return "Pooling"; break;
        case EuclideanLoss: return "EuclideanLoss"; break;
        case Error: return "Error"; break;
    }
    return "Unknown";
}

void BaseLayer::shape() {
    printf("%-17s%-10s %-10s  ", type_to_string().c_str(), name.c_str(), input_name.c_str());
    printf("(%d * %d * %d)\n", info.output_width, info.output_height, info.output_dimension);
//    if (type == LayerType::Convolution || type == LayerType::Fullyconnected) {
//        for (int i = 0; i < info.output_dimension; ++i) {
//            printf("Weight:\n%d: ", i);
//            kernel[i].showWeight();
//        }
//        printf("Bias:\n");
//        biases->showWeight();
//    } else if (type == LayerType::PRelu) {
//        kernel[0].showWeight();
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
        float *bias_weight = biases->getWeight();
        float *bias_grad = biases->getDeltaWeight();
        int output_dimension = info.output_dimension;
        for (int i = 0; i < output_dimension; ++i) {
            Tensor *act_tensor = kernel + i;
            int length = (int)act_tensor->length();
            float *weight = act_tensor->getWeight();
            float *grad = act_tensor->getDeltaWeight();
            for (int j = 0; j < length; j++) {
                weight[j] -= learning_rate * grad[j];
//                if (abs(grad[j]) > 10) {
//                    cout << grad[j];
//                    throw "hello?";
//                }
            }
            bias_weight[i] -= learning_rate * bias_grad[i];
            act_tensor->clearDeltaWeight();
        }
        biases->clearDeltaWeight();
    }
}

void BaseLayer::ClearDeltaWeight() {
    input_tensor->clearDeltaWeight();
}

bool BaseLayer::save(FILE *f) {
    int len = (int)strlen(name.c_str());
    char *name_ = new char [len];
    fwrite(&len, sizeof(int), 1, f);
    strcpy(name_, name.c_str());
    fwrite(name_, sizeof(char), len, f);
    delete [] name_;
    
    len = (int)strlen(input_name.c_str());
    char *input_name_ = new char [len];
    fwrite(&len, sizeof(int), 1, f);
    strcpy(input_name_, input_name.c_str());
    fwrite(input_name_, sizeof(char), len, f);
    delete [] input_name_;
    
    if (type == LayerType::Input) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Fullyconnected) {
        fwrite(&info.input_number, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        for (int i = 0; i < info.output_dimension; ++i) {
            kernel[i].save(f);
        }
        biases->save(f);
    } else if (type == LayerType::Softmax) {
        int input_number = info.input_number / info.output_dimension;
        fwrite(&input_number, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Convolution) {
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        fwrite(&info_more.kernel_width, sizeof(int), 1, f);
        fwrite(&info_more.input_dimension, sizeof(int), 1, f);
        fwrite(&info_more.input_width, sizeof(int), 1, f);
        fwrite(&info_more.input_height, sizeof(int), 1, f);
        fwrite(&info_more.kernel_height, sizeof(int), 1, f);
        fwrite(&info_more.stride, sizeof(int), 1, f);
        fwrite(&info_more.padding, sizeof(int), 1, f);
        for (int i = 0; i < info.output_dimension; ++i) {
            kernel[i].save(f);
        }
        biases->save(f);
    } else if (type == LayerType::Relu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Pooling) {
        fwrite(&info_more.kernel_width, sizeof(int), 1, f);
        fwrite(&info_more.input_dimension, sizeof(int), 1, f);
        fwrite(&info_more.input_width, sizeof(int), 1, f);
        fwrite(&info_more.input_height, sizeof(int), 1, f);
        fwrite(&info_more.kernel_height, sizeof(int), 1, f);
        fwrite(&info_more.stride, sizeof(int), 1, f);
        fwrite(&info_more.padding, sizeof(int), 1, f);
    } else if (type == LayerType::EuclideanLoss) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::PRelu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        kernel->save(f);
    }

    return true;
}

bool BaseLayer::load(FILE *f) {
    if (type == LayerType::Fullyconnected || type == LayerType::Convolution) {
        for (int i = 0; i < info.output_dimension; ++i) {
            kernel[i].load(f);
        }
        biases->load(f);
    } else if (type == LayerType::PRelu) {
        kernel->load(f);
    }
    return true;
}

InputLayer::InputLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Input;
    name = (opt.find("name") == opt.end()) ? "in" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_dimension = (opt.find("input_dimension") == opt.end()) ? atoi(opt["output_dimension"].c_str()) : atoi(opt["input_dimension"].c_str());
    info.output_width = (opt.find("input_width") == opt.end()) ? atoi(opt["output_width"].c_str()) : atoi(opt["input_width"].c_str());
    info.output_height = (opt.find("input_height") == opt.end()) ? atoi(opt["output_height"].c_str()) : atoi(opt["input_height"].c_str());
}

Tensor* InputLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    output_tensor = input_tensor;
    return output_tensor;
}

ConvolutionLayer::ConvolutionLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Convolution;
    name = (opt.find("name") == opt.end()) ? "conv" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_dimension = atoi(opt["number_kernel"].c_str());
    info_more.kernel_width = atoi(opt["kernel_width"].c_str());
    info_more.input_dimension = atoi(opt["input_dimension"].c_str());
    info_more.input_width = atoi(opt["input_width"].c_str());
    info_more.input_height = atoi(opt["input_height"].c_str());
    
    info_more.kernel_height = (opt.find("kernel_height") == opt.end()) ? info_more.kernel_width : atoi(opt["kernel_height"].c_str());
    info_more.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info_more.padding = (opt.find("padding") == opt.end()) ? 0 : atoi(opt["padding"].c_str());
    
    info.output_width = (info_more.input_width + info_more.padding * 2 - info_more.kernel_width) / info_more.stride + 1;
    info.output_height = (info_more.input_height + info_more.padding * 2 - info_more.kernel_height) / info_more.stride + 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(info_more.kernel_width, info_more.kernel_height, info_more.input_dimension);
        kernel[i] = new_kernel;
    }
    biases = new Tensor(1, 1, info.output_dimension, bias);
//    Tensor new_bias(1, 1, info.output_dimension, bias);
//    biases = new_bias;
}

Tensor* ConvolutionLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *act_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0);
    
    int input_width = input_tensor->getWidth();
    int input_height = input_tensor->getHeight();
    int output_width = info.output_width;
    int output_height = info.output_height;
    int xy_stride = info_more.stride;
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
        bias = biases->getWeight();
        offset_h = -info_more.padding;
        kernel_width = act_kernel->getWidth();
        kernel_height = act_kernel->getHeight();
        kernel_dimension = act_kernel->getDimension();
        kernel_weight = act_kernel->getWeight();
        for (output_h = 0; output_h < output_height; ++output_h, offset_h += xy_stride) {
            offset_w = -info_more.padding;
            for (output_w = 0; output_w < output_width; ++output_w, offset_w += xy_stride) {
                sum = 0.0;
                for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    act_h = kernel_h + offset_h;
                    for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < input_height && act_w >= 0 && act_w < input_width) {
                            for (kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
                                sum += kernel_weight[((kernel_width * kernel_h) + kernel_w) * kernel_dimension + kernel_d] * input_weight[((input_width * act_h) + act_w) * info_more.input_dimension + kernel_d];
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
    //input_tensor->clearDeltaWeight();
    int xy_stride = info_more.stride;
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
        bias_grad = biases->getDeltaWeight();
        offset_h = -info_more.padding;
        kernel_width = act_kernel->getWidth();
        kernel_height = act_kernel->getHeight();
        kernel_dimension = act_kernel->getDimension();
        kernel_weight = act_kernel->getWeight();
        kernel_grad = act_kernel->getDeltaWeight();
        for (output_h = 0; output_h < output_height; ++output_h, offset_h += xy_stride) {
            offset_w = -info_more.padding;
            for (output_w = 0; output_w < output_width; ++output_w, offset_w += xy_stride) {
                float chain_grad = output_tensor->getGrad(output_w, output_h, output_d);
                for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    act_h = kernel_h + offset_h;
                    for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        act_w = kernel_w + offset_w;
                        if (act_h >= 0 && act_h < info_more.input_height && act_w >= 0 && act_w < info_more.input_width) {
                            for (kernel_d = 0; kernel_d < kernel_dimension; ++kernel_d) {
                                index_1 = ((info_more.input_width * output_h) + output_w) * info_more.input_dimension + kernel_d;
                                index_2 = ((kernel_width * kernel_h) + kernel_w) * kernel_dimension + kernel_d;
                                kernel_grad[index_2] += input_weight[index_1] * chain_grad;
                                input_grad[index_1] += kernel_weight[index_2] * chain_grad;
//                                if (abs(chain_grad) > 10) {
//                                    cout << chain_grad;
//                                    throw "WTF";
//                                }
//                                if (abs(kernel_grad[index_2]) > 10) {
//                                    cout << kernel_grad[index_2];
//                                    throw "WTF";
//                                }
//                                if (abs(input_grad[index_1]) > 10) {
//                                    cout << input_grad[index_1];
//                                    throw "WTF";
//                                }
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
    type = LayerType::Pooling;
    name = (opt.find("name") == opt.end()) ? "pool" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info_more.kernel_width = atoi(opt["kernel_width"].c_str());
    info_more.input_dimension = atoi(opt["input_dimension"].c_str());
    info_more.input_width = atoi(opt["input_width"].c_str());
    info_more.input_height = atoi(opt["input_height"].c_str());
    
    info_more.kernel_height = (opt.find("kernel_height") == opt.end()) ? info_more.kernel_width : atoi(opt["kernel_height"].c_str());
    info_more.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info_more.padding = (opt.find("padding") == opt.end()) ? 0 : atoi(opt["padding"].c_str());
    
    info.output_dimension = info_more.input_dimension;
    info.output_width = (info_more.input_width + info_more.padding * 2 - info_more.kernel_width) / info_more.stride + 1;
    info.output_height = (info_more.input_height + info_more.padding * 2 - info_more.kernel_height) / info_more.stride + 1;
    
    choosex.assign(info.output_width * info.output_height * info.output_dimension, 0);
    choosey.assign(info.output_width * info.output_height * info.output_dimension, 0);
}

Tensor* PoolingLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
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
        offset_w = -info_more.padding;
        for (output_w = 0; output_w < output_width; ++output_w, offset_w += info_more.stride) {
            offset_h = -info_more.padding;
            for (output_h = 0; output_h < output_height; ++output_h, offset_h += info_more.stride) {
                minimum = -100000000;
                win_x = -1;
                win_y = -1;
                for (kernel_w = 0; kernel_w < info_more.kernel_width; ++kernel_w) {
                    for (kernel_h = 0; kernel_h < info_more.kernel_height; ++kernel_h) {
                        act_w = offset_w + output_w;
                        act_h = offset_h + output_h;
                        if (act_w >= 0 && act_w < info_more.input_width && act_h >= 0 && act_h < info_more.input_height) {
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
    //input->clearDeltaWeight();
    Tensor *output = output_tensor;
    
    int output_dimension = info.output_dimension;
    int output_width = info.output_width;
    int output_height = info.output_height;
    int counter = 0;
    int cal_d, cal_w, cal_h;
    int offset_w, offset_h;
    float chain_grad;
    
    for (cal_d = 0; cal_d < output_dimension; ++cal_d) {
        offset_w = -info_more.padding;
        for (cal_w = 0; cal_w < output_width; ++cal_w, offset_w += info_more.stride) {
            offset_h = -info_more.padding;
            for (cal_h = 0; cal_h < output_height; ++cal_h, offset_h += info_more.stride) {
                chain_grad = output->getGrad(cal_w, cal_h, cal_d);
                input->addGrad(choosex[counter], choosey[counter], cal_d, chain_grad);
                counter++;
//                if (abs(chain_grad) > 10) {
//                    cout << chain_grad;
//                    throw "WTF";
//                }
            }
        }
    }
}

FullyConnectedLayer::FullyConnectedLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Fullyconnected;
    name = (opt.find("name") == opt.end()) ? "fc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
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
    biases = new Tensor(1, 1, info.output_dimension, bias);
//    Tensor new_bias(1, 1, info.output_dimension, bias);
//    biases = new_bias;
}

Tensor* FullyConnectedLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(1, 1, info.output_dimension, 0);
    float *pos = cal->getWeight();
    float *input = input_tensor_->getWeight();
    float *bias = biases->getWeight();
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
    //cal->clearDeltaWeight();
    float *cal_w = cal->getWeight();
    float *cal_dw = cal->getDeltaWeight();
    float *act_biases_grad = biases->getDeltaWeight();
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
    type = LayerType::Relu;
    name = (opt.find("name") == opt.end()) ? "relu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
}

Tensor* ReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(input_tensor_);
    float *val = cal->getWeight();
    for (int i = 0; i < info.input_number; ++i) {
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
    //cal->clearDeltaWeight();
    float *act_weight = output_tensor->getWeight();
    float *act_grad = output_tensor->getDeltaWeight();
    float *pos_grad = cal->getDeltaWeight();
    
    for (int i = 0; i < info.input_number; ++i) {
        if (act_weight[i] <= 0)
            pos_grad[i] = 0;
        else
            pos_grad[i] += act_grad[i];
    }
}

PReluLayer::PReluLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::PRelu;
    name = (opt.find("name") == opt.end()) ? "prelu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.input_number, 0.25);
}

Tensor* PReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    Tensor *cal = new Tensor(input_tensor_);
    float *val = cal->getWeight();
    float *alpha = kernel->getWeight();
    for (int i = 0; i < info.input_number; ++i) {
        if (val[i] < 0)
            val[i] *= alpha[i];
    }
    if (output_tensor)
        delete output_tensor;
    output_tensor = cal;
    return output_tensor;
}

void PReluLayer::Backward() {
    Tensor *cal = input_tensor;
    //cal->clearDeltaWeight();
    float *act_weight = output_tensor->getWeight();
    float *act_grad = output_tensor->getDeltaWeight();
    float *pos_grad = cal->getDeltaWeight();
    float *alpha = kernel->getWeight();
    float *alpha_grad = kernel->getDeltaWeight();
    
    for (int i = 0; i < info.input_number; ++i) {
        if (act_weight[i] <= 0) {
            pos_grad[i] = act_grad[i] * alpha[i];
            alpha_grad[i] = act_weight[i] * act_grad[i];
        }
        else {
            pos_grad[i] = act_grad[i];
            alpha_grad[i] = 0;
        }
    }
}

SoftmaxLayer::SoftmaxLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Softmax;
    name = (opt.find("name") == opt.end()) ? "sm" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
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

float SoftmaxLayer::Backward(vfloat& target) {
    Tensor *cal_tensor = input_tensor;
    cal_tensor->clearDeltaWeight();
    float *cal_delta_weight = cal_tensor->getDeltaWeight();
    int output_dimension = info.output_dimension;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indicator = (i == target[0]) ? 1.0 : 0.0;
        float mul = -(indicator - expo_sum[i]);
        cal_delta_weight[i] = mul;
        if (abs(cal_delta_weight[i]) > 10) {
            cout << cal_delta_weight[i];
            throw "WTF";
        }
    }
    return -log(expo_sum[(int)target[0]]);
}

EuclideanLossLayer::EuclideanLossLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::EuclideanLoss;
    name = (opt.find("name") == opt.end()) ? "el" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_dimension = atoi(opt["input_dimension"].c_str()) * atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str());
    info.output_width = 1;
    info.output_height = 1;
}

Tensor* EuclideanLossLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    output_tensor = input_tensor;
    return output_tensor;
}

float EuclideanLossLayer::Backward(vfloat& target) {
    Tensor *cal_tensor = input_tensor;
    cal_tensor->clearDeltaWeight();
    float *cal_weight = cal_tensor->getWeight();
    float *cal_delta_weight = cal_tensor->getDeltaWeight();
    int output_dimension = info.output_dimension;
    float loss = 0;
    
    for (int i = 0; i < output_dimension; ++i) {
        float delta = cal_weight[i] - target[i];
        cal_delta_weight[i] = delta;
        loss += 0.5 * delta * delta;
    }
    return loss;
}

