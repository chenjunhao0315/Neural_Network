//
//  Layer.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Layer.hpp"

Model_Layer::~Model_Layer() {
    switch (type) {
        case LayerType::Input: delete input_layer; break;
        case LayerType::Fullyconnected: delete fullyconnected_layer; break;
        case LayerType::Relu: delete relu_layer; break;
        case LayerType::Softmax: delete softmax_layer; break;
        case LayerType::Convolution: delete convolution_layer; break;
        case LayerType::Pooling: delete pooling_layer; break;
        case LayerType::EuclideanLoss: delete euclideanloss_layer; break;
        case LayerType::PRelu: delete prelu_layer; break;
        case LayerType::ShortCut: delete shortcut_layer; break;
        default: break;
    }
    input_layer = nullptr;
    fullyconnected_layer = nullptr;
    relu_layer = nullptr;
    prelu_layer = nullptr;
    softmax_layer = nullptr;
    convolution_layer = nullptr;
    pooling_layer = nullptr;
    euclideanloss_layer = nullptr;
    shortcut_layer = nullptr;
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
    shortcut_layer = nullptr;
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
    shortcut_layer = nullptr;
    if (this != &L) {
        type = L.type;
        switch (type) {
            case LayerType::Input:
                input_layer = new InputLayer(*L.input_layer);
                break;
            case LayerType::Fullyconnected:
                fullyconnected_layer = new FullyConnectedLayer(*L.fullyconnected_layer);
                break;
            case LayerType::Relu:
                relu_layer = new ReluLayer(*L.relu_layer);
                break;
            case LayerType::Softmax:
                softmax_layer = new SoftmaxLayer(*L.softmax_layer);
                break;
            case LayerType::Convolution:
                convolution_layer = new ConvolutionLayer(*L.convolution_layer);
                break;
            case LayerType::Pooling:
                pooling_layer = new PoolingLayer(*L.pooling_layer);
                break;
            case LayerType::EuclideanLoss:
                euclideanloss_layer = new EuclideanLossLayer(*L.euclideanloss_layer);
                break;
            case LayerType::PRelu:
                prelu_layer = new PReluLayer(*L.prelu_layer);
                break;
            case LayerType::ShortCut:
                shortcut_layer = new ShortCutLayer(*L.shortcut_layer);
                break;
            default:
                break;
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
    shortcut_layer = L.shortcut_layer;
    L.input_layer = nullptr;
    L.fullyconnected_layer = nullptr;
    L.relu_layer = nullptr;
    L.prelu_layer = nullptr;
    L.softmax_layer = nullptr;
    L.convolution_layer = nullptr;
    L.pooling_layer = nullptr;
    L.euclideanloss_layer = nullptr;
    L.shortcut_layer = nullptr;
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
    shortcut_layer = nullptr;
    if (this != &L) {
        type = L.type;
        switch (type) {
            case LayerType::Input:
                input_layer = new InputLayer(*L.input_layer);
                break;
            case LayerType::Fullyconnected:
                fullyconnected_layer = new FullyConnectedLayer(*L.fullyconnected_layer);
                break;
            case LayerType::Relu:
                relu_layer = new ReluLayer(*L.relu_layer);
                break;
            case LayerType::Softmax:
                softmax_layer = new SoftmaxLayer(*L.softmax_layer);
                break;
            case LayerType::Convolution:
                convolution_layer = new ConvolutionLayer(*L.convolution_layer);
                break;
            case LayerType::Pooling:
                pooling_layer = new PoolingLayer(*L.pooling_layer);
                break;
            case LayerType::EuclideanLoss:
                euclideanloss_layer = new EuclideanLossLayer(*L.euclideanloss_layer);
                break;
            case LayerType::PRelu:
                prelu_layer = new PReluLayer(*L.prelu_layer);
                break;
            case LayerType::ShortCut:
                shortcut_layer = new ShortCutLayer(*L.shortcut_layer);
                break;
            default:
                break;
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
    shortcut_layer = nullptr;
    type = string_to_type(opt_["type"]);
    
    switch (type) {
        case LayerType::Input:
            input_layer = new InputLayer(opt_);
            break;
        case LayerType::Fullyconnected:
            fullyconnected_layer = new FullyConnectedLayer(opt_);
            break;
        case LayerType::Relu:
            relu_layer = new ReluLayer(opt_);
            break;
        case LayerType::Softmax:
            softmax_layer = new SoftmaxLayer(opt_);
            break;
        case LayerType::Convolution:
            convolution_layer = new ConvolutionLayer(opt_);
            break;
        case LayerType::Pooling:
            pooling_layer = new PoolingLayer(opt_);
            break;
        case LayerType::EuclideanLoss:
            euclideanloss_layer = new EuclideanLossLayer(opt_);
            break;
        case LayerType::PRelu:
            prelu_layer = new PReluLayer(opt_);
            break;
        case LayerType::ShortCut:
            shortcut_layer = new ShortCutLayer(opt_);
            break;
        default:
            break;
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
    } else if (type == "ShortCut") {
        return LayerType::ShortCut;
    }
    return LayerType::Error;
}

Tensor* Model_Layer::Forward(Tensor* input_tensor_, Tensor* shortcut_tensor_) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->Forward(input_tensor_);
            break;
        case LayerType::Pooling:
            return pooling_layer->Forward(input_tensor_);
            break;
        case LayerType::PRelu:
            return prelu_layer->Forward(input_tensor_);
            break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->Forward(input_tensor_);
            break;
        case LayerType::Relu:
            return relu_layer->Forward(input_tensor_);
            break;
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Forward(input_tensor_);
            break;
        case LayerType::Softmax:
            return softmax_layer->Forward(input_tensor_);
            break;
        case LayerType::Input:
            return input_layer->Forward(input_tensor_);
            break;
        case LayerType::ShortCut:
            return shortcut_layer->Forward(input_tensor_, shortcut_tensor_);
        default:
            break;
    }
    return nullptr;
}

float Model_Layer::Backward(vfloat& target) {
    switch (type) {
        case LayerType::Softmax:
            return softmax_layer->Backward(target);
            break;
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Backward(target);
            break;
        default:
            break;
    }
    return 0;
}

void Model_Layer::Backward() {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->Backward();
            break;
        case LayerType::Pooling:
            return pooling_layer->Backward();
            break;
        case LayerType::PRelu:
            return prelu_layer->Backward();
            break;
        case LayerType::Input:
            return input_layer->Backward();
            break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->Backward();
            break;
        case LayerType::Relu:
            return relu_layer->Backward();
            break;
        case LayerType::ShortCut:
            return shortcut_layer->Backward();
            break;
        default:
            break;
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
    } else if (type == LayerType::ShortCut) {
        shortcut_layer->shape();
    }
}

int Model_Layer::getParameter(int type_) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->getParameter(type_);
            break;
        case LayerType::Pooling:
            return pooling_layer->getParameter(type_);
            break;
        case LayerType::PRelu:
            return prelu_layer->getParameter(type_);
            break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getParameter(type_);
            break;
        case LayerType::Relu:
            return relu_layer->getParameter(type_);
            break;
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->getParameter(type_);
            break;
        case LayerType::Softmax:
            return softmax_layer->getParameter(type_);
            break;
        case LayerType::Input:
            return input_layer->getParameter(type_);
            break;
        case LayerType::ShortCut:
            return shortcut_layer->getParameter(type_);
            break;
        default:
            break;
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
    } else if (type == LayerType::ShortCut) {
        return shortcut_layer->save(f);
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
    } else if (type == LayerType::ShortCut) {
        return shortcut_layer->load(f);
    }
    return 0;
}

void Model_Layer::Update() {
    if (type == LayerType::Fullyconnected) {
        fullyconnected_layer->Update();
    } else if (type == LayerType::Convolution) {
        convolution_layer->Update();
    } else if (type == LayerType::PRelu) {
        prelu_layer->Update();
    }
}

void Model_Layer::ClearGrad() {
    if (type == LayerType::Input) {
        input_layer->ClearGrad();
    } else if (type == LayerType::Fullyconnected) {
        fullyconnected_layer->ClearGrad();
    } else if (type == LayerType::Relu) {
        relu_layer->ClearGrad();
    } else if (type == LayerType::Softmax) {
        softmax_layer->ClearGrad();
    } else if (type == LayerType::Convolution) {
        convolution_layer->ClearGrad();
    } else if (type == LayerType::Pooling) {
        pooling_layer->ClearGrad();
    } else if (type == LayerType::EuclideanLoss) {
        euclideanloss_layer->ClearGrad();
    } else if (type == LayerType::PRelu) {
        prelu_layer->ClearGrad();
    } else if (type == LayerType::ShortCut) {
        shortcut_layer->ClearGrad();
    }
}

Tensor* Model_Layer::getKernel() {
    switch(type) {
        case LayerType::Convolution:
            return convolution_layer->getKernel(); break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getKernel(); break;
        case LayerType::PRelu:
            return prelu_layer->getKernel(); break;
        default:
            break;
    }
    return nullptr;
}

Tensor* Model_Layer::getBiases() {
    switch(type) {
        case LayerType::Convolution:
            return convolution_layer->getBiases(); break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getBiases(); break;
        case LayerType::PRelu:
            return prelu_layer->getKernel(); break;
        default:
            break;
    }
    return nullptr;
}

vfloat Model_Layer::getDetailParameter() {
    switch(type) {
        case LayerType::Convolution:
            return convolution_layer->getDetailParameter(); break;
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getDetailParameter(); break;
        case LayerType::PRelu:
            return prelu_layer->getDetailParameter(); break;
        default:
            break;
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
    detail.reserve(8);
    switch(type) {
        case LayerType::Convolution:
            detail.push_back(info.output_dimension);
            detail.push_back(info_more.kernel_width);
            detail.push_back(info_more.kernel_height);
            detail.push_back(info_more.input_dimension);
            detail.push_back(info.output_dimension);
            detail.push_back(0);    // l1 decay mul
            detail.push_back(1);    // l2 decay mul
            break;
        case LayerType::Fullyconnected:
            detail.push_back(info.output_dimension);
            detail.push_back(1);
            detail.push_back(1);
            detail.push_back(info.input_number);
            detail.push_back(info.output_dimension);
            detail.push_back(0);    // l1 decay mul
            detail.push_back(1);    // l2 decay mul
            break;
        case LayerType::PRelu:
            detail.push_back(1);
            detail.push_back(1);
            detail.push_back(1);
            detail.push_back(info.input_number);
            detail.push_back(0);
            detail.push_back(0);    // l1 decay mul
            detail.push_back(0);    // l2 decay mul
            break;
        default:
            break;
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
    if (workspace)
        delete [] workspace;
    if (weight_index)
        delete [] weight_index;
    if (input_index)
        delete [] input_index;
}

BaseLayer::BaseLayer() {
    input_tensor = nullptr;
    output_tensor = nullptr;
    kernel = nullptr;
    biases = nullptr;
    workspace = nullptr;
    weight_index = nullptr;
    input_index = nullptr;
}

BaseLayer::BaseLayer(BaseLayer *L) {
    printf("address copy\n");
    if (this != L) {
        type = L->type;
        name = L->name;
        input_name = L->input_name;
        info = L->info;
        info_more = L->info_more;
        opt = L->opt;
        if (L->input_tensor)
            input_tensor = new Tensor(L->input_tensor);
        else
            input_tensor = nullptr;
        if (L->output_tensor)
            output_tensor = new Tensor(L->output_tensor);
        else
            output_tensor = nullptr;
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
        if (L->workspace) {
            workspace = new float [info_more.kernel_width * info_more.kernel_height * info_more.input_dimension * info.output_width * info.output_height];
            //            cout << "this: " << this << " address copy workspace: " << workspace << endl;
        } else {
            workspace = nullptr;
        }
        if (L->weight_index) {
            weight_index = new int [info_more.weight_index_size];
            memcpy(weight_index, L->weight_index, sizeof(int) * info_more.weight_index_size);
        } else {
            weight_index = nullptr;
        }
        if (L->input_index) {
            input_index = new int [info_more.input_index_size];
            memcpy(input_index, L->input_index, sizeof(int) * info_more.input_index_size);
        } else {
            input_index = nullptr;
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
        if (L.input_tensor)
            input_tensor = new Tensor(L.input_tensor);
        else
            input_tensor = nullptr;
        if (L.output_tensor)
            output_tensor = new Tensor(L.output_tensor);
        else
            output_tensor = nullptr;
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
        if (L.workspace) {
            workspace = new float [info_more.kernel_width * info_more.kernel_height * info_more.input_dimension * info.output_width * info.output_height];
            //            cout << "this: " << this << " copy workspace: " << workspace << endl;
        } else {
            workspace = nullptr;
        }
        if (L.weight_index) {
            weight_index = new int [info_more.weight_index_size];
            memcpy(weight_index, L.weight_index, sizeof(int) * info_more.weight_index_size);
        } else {
            weight_index = nullptr;
        }
        if (L.input_index) {
            input_index = new int [info_more.input_index_size];
            memcpy(input_index, L.input_index, sizeof(int) * info_more.input_index_size);
        } else {
            input_index = nullptr;
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
    workspace = L.workspace;
    L.workspace = nullptr;
    weight_index = L.weight_index;
    L.weight_index = nullptr;
    input_index = L.input_index;
    L.input_index = nullptr;
    //    cout << "move workspace: " << workspace << endl;
}

BaseLayer& BaseLayer::operator=(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        info_more = L.info_more;
        opt = L.opt;
        if (L.input_tensor)
            input_tensor = new Tensor(L.input_tensor);
        else
            input_tensor = nullptr;
        if (L.output_tensor)
            output_tensor = new Tensor(L.output_tensor);
        else
            output_tensor = nullptr;
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
        if (L.workspace) {
            workspace = new float [info_more.kernel_width * info_more.kernel_height * info_more.input_dimension * info.output_width * info.output_height];
            cout << "assign workspace: " << workspace << endl;
        } else {
            workspace = nullptr;
        }
        if (L.weight_index) {
            weight_index = new int [info_more.weight_index_size];
            memcpy(weight_index, L.weight_index, sizeof(int) * info_more.weight_index_size);
        } else {
            weight_index = nullptr;
        }
        if (L.input_index) {
            input_index = new int [info_more.input_index_size];
            memcpy(input_index, L.input_index, sizeof(int) * info_more.input_index_size);
        } else {
            input_index = nullptr;
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
        case ShortCut: return "ShortCut"; break;
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
        float *bias_weight = biases->weight;
        float *bias_grad = biases->delta_weight;
        int output_dimension = info.output_dimension;
        for (int i = 0; i < output_dimension; ++i) {
            Tensor *act_tensor = kernel + i;
            int length = (int)act_tensor->length();
            float *weight = act_tensor->weight;
            float *grad = act_tensor->delta_weight;
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

void BaseLayer::Update() {
    if (type == LayerType::Fullyconnected || type == LayerType::Convolution) {
        float *bias_weight = biases->weight;
        float *bias_grad = biases->delta_weight;
        int output_dimension = info.output_dimension;
        for (int i = 0; i < output_dimension; ++i) {
            Tensor *act_tensor = kernel + i;
            int length = (int)act_tensor->length();
            float *weight = act_tensor->weight;
            float *grad = act_tensor->delta_weight;
            for (int j = 0; j < length; j++) {
                weight[j] += grad[j];
            }
            bias_weight[i] += bias_grad[i];
            act_tensor->clearDeltaWeight();
        }
        biases->clearDeltaWeight();
    } else if (type == LayerType::PRelu) {
        float *weight = kernel[0].getWeight();
        float *grad = kernel[0].getDeltaWeight();
        for (int i = 0; i < info.input_number; ++i) {
            weight[i] += grad[i];
        }
        kernel[0].clearDeltaWeight();
    }
}

void BaseLayer::ClearGrad() {
//    input_tensor->clearDeltaWeight();
//    output_tensor->clearDeltaWeight();
//    if (type == LayerType::Convolution || type == LayerType::Fullyconnected) {
//        for (int i = 0; i < info.output_dimension; ++i) {
//            kernel[i].clearDeltaWeight();
//        }
//        biases->clearDeltaWeight();
//    } else if (type == LayerType::PRelu) {
//        kernel[0].clearDeltaWeight();
//    }
    output_tensor->clearDeltaWeight();
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
    } else if (type == LayerType::ShortCut) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        int len = (int)strlen(opt["shortcut"].c_str());
        fwrite(&len, sizeof(int), 1, f);
        char *output = new char [len];
        strcpy(output, opt["shortcut"].c_str());
        fwrite(output, sizeof(char), len, f);
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
    info_more.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info_more.kernel_width - 1) / 2) : atoi(opt["padding"].c_str()));
    
    info.output_width = (info_more.input_width + info_more.padding * 2 - info_more.kernel_width) / info_more.stride + 1;
    info.output_height = (info_more.input_height + info_more.padding * 2 - info_more.kernel_height) / info_more.stride + 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(info_more.kernel_width, info_more.kernel_height, info_more.input_dimension);
        kernel[i] = new_kernel;
    }
    biases = new Tensor(1, 1, info.output_dimension, bias);
    workspace = new float [info_more.kernel_width * info_more.kernel_height * info_more.input_dimension * info.output_width * info.output_height];
    
    int input_width = info_more.input_width;
    int input_height = info_more.input_height;
    int stride = info_more.stride;
    int neg_padding = -info_more.padding;
    int out_dim, out_height, out_width;
    int x, y;
    int kernel_w, kernel_h, kernel_dim;
    int output_dimension = info.output_dimension;
    int output_height = info.output_height;
    int output_width = info.output_width;
    int coordinate_x, coordinate_y;
    int kernel_width = info_more.kernel_width;
    int kernel_height = info_more.kernel_height;
    int input_dimension = info_more.input_dimension;
    int kernel_n = kernel_width * kernel_height * input_dimension;
    int out_size = output_width * output_height;
    int workspace_size = out_size * kernel_n;
    int count;
    
    info_more.workspace_size = workspace_size;
    info_more.input_index_size = workspace_size;
    info_more.weight_index_size = out_size * output_dimension;
    
    input_index = new int [workspace_size];
    fill(input_index, input_index + workspace_size, -1);
    count = -1;
    y = neg_padding;
    for (out_height = 0; out_height < output_height; y += stride, ++out_height) {
        x = neg_padding;
        for (out_width = 0; out_width < output_width; x += stride, ++out_width)
        for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
            coordinate_y = y + kernel_h;
            for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                coordinate_x = x + kernel_w;
                if (coordinate_y >= 0 && coordinate_y < input_height && coordinate_x >= 0 && coordinate_x < input_width)
                    for (kernel_dim = 0; kernel_dim < input_dimension; ++kernel_dim)
                input_index[++count] = ((input_width * coordinate_y) + coordinate_x) * input_dimension + kernel_dim;
                else
                    count += input_dimension;
            }
        }
    }
    
    weight_index = new int [out_size * output_dimension];
    count = -1;
    for (out_dim = 0; out_dim < output_dimension; ++out_dim)
    for (out_height = 0; out_height < output_height; ++out_height)
    for (out_width = 0; out_width < output_width; ++out_width)
    weight_index[++count] = ((out_height * output_width) + out_width) * output_dimension + out_dim;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0.0);
}

Tensor* ConvolutionLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    output_tensor->clearDeltaWeight();
    
    float *input_weight = input_tensor->weight;
    float *bias_ptr = biases->weight;
    
    int kernel_n = info_more.kernel_width * info_more.kernel_height * info_more.input_dimension;
    int out_size = info.output_width * info.output_height;
    
    int count;
    int index;
    for (count = info_more.workspace_size; count--; ) {
        workspace[count] = ((index = input_index[count]) == -1) ? 0 : input_weight[index];
    }
    
    float conv, bias;
    float *kernel_weight, *kernel_weight_ptr;
    float *output_weight = output_tensor->weight;
    int *output_index_ptr = weight_index;
    float *src;
    Tensor *kernel_ptr = kernel;
    for (int out_dim = info.output_dimension; out_dim--; ) {
        src = workspace;
        kernel_weight = (kernel_ptr++)->weight;
        bias = *(bias_ptr++);
        for (index = out_size; index--; ) {
            conv = 0.0f;
            kernel_weight_ptr = kernel_weight;
            for (count = kernel_n; count--; )
            conv += *(kernel_weight_ptr++) * *(src++);
            output_weight[*(output_index_ptr++)] = conv + bias;
        }
    }
    return output_tensor;
}

void ConvolutionLayer::Backward() {
    float *input_weight = input_tensor->weight;
    float *input_grad = input_tensor->delta_weight;
    float *bias_grad = biases->delta_weight;
    
    float *kernel_weight, *kernel_grad;
    int kernel_n = info_more.kernel_width * info_more.kernel_height * info_more.input_dimension;
    
    int *input_index_ptr;
    
    float chain_grad;
    float *output_grad = output_tensor->delta_weight;
    int out_size = info.output_width * info.output_height;
    
    int index;
    int out_index;
    int kernel_index;
    int weight_intex_ptr = -1;
    for (int out_dim = 0; out_dim < info.output_dimension; ++out_dim) {
        Tensor *kernel_act = kernel + out_dim;
        kernel_weight = kernel_act->weight;
        kernel_grad = kernel_act->delta_weight;
        input_index_ptr = input_index;
        
        for (out_index = out_size; out_index; --out_index) {
            chain_grad = output_grad[weight_index[++weight_intex_ptr]];
            for (kernel_index = 0; kernel_index < kernel_n; ++kernel_index) {
                if ((index = *(input_index_ptr++)) != -1) {
                    kernel_grad[kernel_index] += input_weight[index] * chain_grad;
                    input_grad[index] += kernel_weight[kernel_index] * chain_grad;
                }
            }
            bias_grad[out_dim] += chain_grad;
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
    
    int output_dimension = info.output_dimension;
    int output_width = info.output_width;
    int output_height = info.output_height;
    int kernel_width = info_more.kernel_width;
    int kernel_height = info_more.kernel_height;
    int input_width = info_more.input_width;
    int input_height = info_more.input_height;
    int input_dimension = info_more.input_dimension;
    int stride = info_more.stride;
    int neg_padding = -info_more.padding;
    
    input_index = new int [output_dimension * output_width * output_height * kernel_width * kernel_height];
    fill(input_index, input_index + output_dimension * output_width * output_height * kernel_width * kernel_height, -1);
    weight_index = new int [output_dimension * output_width * output_height];
    
    int input_counter = 0;
    int weight_counter = 0;
    
    info_more.input_index_size = output_dimension * output_width * output_height * kernel_width * kernel_height;
    info_more.weight_index_size =output_dimension * output_width * output_height;
    
    for (int output_d = 0; output_d < output_dimension; ++output_d) {
        int offset_w = neg_padding;
        for (int output_w = 0; output_w < output_width; ++output_w, offset_w += stride) {
            int offset_h = neg_padding;
            for (int output_h = 0; output_h < output_height; ++output_h, offset_h += stride) {
                for (int kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                    int act_w = offset_w + kernel_w;
                    for (int kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                        int act_h = offset_h + kernel_h;
                        if (act_w >= 0 && act_w < input_width && act_h >= 0 && act_h < input_height) {
                            input_index[input_counter] = ((act_h * input_width) + act_w) * input_dimension + output_d;
                        }
                        ++input_counter;
                    }
                }
                weight_index[weight_counter++] = ((output_h * output_width) + output_w) * output_dimension + output_d;
            }
        }
    }
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0.0);
}

Tensor* PoolingLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    int output_width = info.output_width;
    int output_height = info.output_height;
    int output_dimension = info.output_dimension;
    int output_w, output_h;
    int offset_w, offset_h;
    float maximum;
    int win_x = -1, win_y = -1;
    int kernel_w, kernel_h;
    float value = 0.0;
    int neg_padding = -info_more.padding;
    int stride = info_more.stride;
    int kernel_width = info_more.kernel_width;
    int kernel_height = info_more.kernel_height;
    
    int counter = 0;
    float *input_weight = input_tensor->weight;
    float *output_weight = output_tensor->weight;
    int *input_index_ptr = input_index;
    int *weight_index_ptr = weight_index;
    int index;
    for (int output_d = 0; output_d < output_dimension; ++output_d) {
        offset_w = neg_padding;
        for (output_w = 0; output_w < output_width; ++output_w, offset_w += stride) {
            offset_h = neg_padding;
            for (output_h = 0; output_h < output_height; ++output_h, offset_h += stride) {
                maximum = -100000000;
                for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                    for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                        if ((index = *(input_index_ptr++)) != -1) {
                            value = input_weight[index];
                            if (value > maximum) {
                                maximum = value;
                                win_x = offset_w + kernel_w;
                                win_y = offset_h + kernel_h;
                            }
                        }
                    }
                }
                choosex[counter] = win_x;
                choosey[counter++] = win_y;
                output_weight[*(weight_index_ptr++)] = maximum;
            }
        }
    }

    return output_tensor;
}

void PoolingLayer::Backward() {
    Tensor *input = input_tensor;

    int counter = 0;
    float chain_grad;
    float *output_grad = output_tensor->delta_weight;
    int *weight_index_ptr = weight_index;

    int out_size = info.output_width * info.output_height;

    for (int cal_d = 0; cal_d < info.output_dimension; ++cal_d) {
        for (int out_counter = 0; out_counter < out_size; ++out_counter) {
            chain_grad = output_grad[*(weight_index_ptr++)];
            input->addGrad(choosex[counter], choosey[counter], cal_d, chain_grad);
            ++counter;
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
    output_tensor = new Tensor(1, 1, info.output_dimension, 0);
}

Tensor* FullyConnectedLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    float *pos = output_tensor->weight;
    float *input = input_tensor_->weight;
    float *bias = biases->weight;
    
    float sum;
    float *weight;
    
    int i, j;
    float *input_ptr;
    // get output
    for (i = info.output_dimension; i--; ) {
        sum = 0.0;
        weight = kernel[i].weight;
        input_ptr = input;
        for (j = info.input_number; j; --j) {
            sum += *(input_ptr++) * *(weight++);
        }
        
        pos[i] = sum + bias[i];
    }
    return output_tensor;
}

void FullyConnectedLayer::Backward() {
    Tensor *cal = input_tensor;
    float *cal_w = cal->weight;
    float *cal_dw = cal->delta_weight;
    float *act_biases_grad = biases->delta_weight;
    float *output_grad = output_tensor->delta_weight;
    
    float *act_weight, *act_grad;
    float chain_grad;
    
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor *kernel_act = kernel + i;
        act_weight = kernel_act->weight;
        act_grad = kernel_act->delta_weight;
        chain_grad = output_grad[i];
        for (int j = 0; j < info.input_number; ++j) {
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
    
    output_tensor = new Tensor(info.output_width, info.output_height,info.output_dimension, 0.0);
}

Tensor* ReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
//    Tensor *cal = new Tensor(input_tensor_);
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    for (int i = 0; i < info.input_number; ++i) {
//        if (val[i] < 0)
//            val[i] = 0;
        out[i] = (val[i] < 0) ? 0 : val[i];
    }
    //if (output_tensor)
//    delete output_tensor;
//    output_tensor = cal;
    return output_tensor;
}

void ReluLayer::Backward() {
    Tensor *cal = input_tensor;
    //cal->clearDeltaWeight();
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = cal->delta_weight;
    
    for (int i = 0; i < info.input_number; ++i) {
        if (act_weight[i] <= 0)
            pos_grad[i] += 0;
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
    float alpha = (opt.find("alpha") == opt.end()) ? 0.25 : atof(opt["alpha"].c_str());
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension, alpha);
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension, 0.0);
}

Tensor* PReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    float *alpha = kernel->weight;
    float value;
    for (int i = info.input_number; i--; alpha++) {
        value = *(val++);
        *(out++) = (value < 0) ? value * *alpha : value;
    }
    
    return output_tensor;
}

void PReluLayer::Backward() {
    Tensor *cal = input_tensor;
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = cal->delta_weight;
    float *alpha = kernel->weight;
    float *alpha_grad = kernel->delta_weight;
    float chain_grad;
    
    for (int i = 0; i < info.input_number; ++i) {
        chain_grad = act_grad[i];
        if (act_weight[i] < 0) {
            pos_grad[i] += chain_grad * alpha[i];
            alpha_grad[i] = act_weight[i] * chain_grad;
        }
        else {
            pos_grad[i] += chain_grad;
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
    
    output_tensor = new Tensor(1, 1, info.output_dimension, 0);
}

Tensor* SoftmaxLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    float *act = input_tensor_->weight;
    float max = act[0];
    for (int i = 1; i < info.output_dimension; ++i) {
        if (act[i] > max)
            max = act[i];
    }
    if (!expo_sum)
        expo_sum = new float [info.output_dimension];
    
    float sum = 0;
    for (int i = 0; i < info.output_dimension; ++i) {
        sum += (expo_sum[i] = exp(act[i] - max));
    }
    
    float *cal_weight = output_tensor->weight;
    for (int i = 0; i < info.output_dimension; ++i) {
        expo_sum[i] /= sum;
        cal_weight[i] = expo_sum[i];
    }
    
    return output_tensor;
}

float SoftmaxLayer::Backward(vfloat& target) {
    Tensor *cal_tensor = input_tensor;
    cal_tensor->clearDeltaWeight();
    float *cal_delta_weight = cal_tensor->delta_weight;
    int output_dimension = info.output_dimension;
    
    for (int i = 0; i < output_dimension; ++i) {
        float indicator = (i == target[0]) ? 1.0 : 0.0;
        float mul = -(indicator - expo_sum[i]);
        cal_delta_weight[i] = mul;
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
    float *cal_weight = cal_tensor->weight;
    float *cal_delta_weight = cal_tensor->delta_weight;
    float loss = 0;
    
    for (int i = 0; i < info.output_dimension; ++i) {
        float delta = cal_weight[i] - target[i];
        cal_delta_weight[i] = delta;
        loss += 0.5 * delta * delta;
    }
    return loss;
}

ShortCutLayer::ShortCutLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::ShortCut;
    name = (opt.find("name") == opt.end()) ? "sc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
}

Tensor* ShortCutLayer::Forward(Tensor *input_tensor_, Tensor *shortcut_tensor_) {
    input_tensor = input_tensor_;
    shortcut_tensor = shortcut_tensor_;
    
    Tensor *cal = new Tensor(input_tensor_);
    int w1 = shortcut_tensor->width;
    int w2 = info.output_width;
    int h1 = shortcut_tensor->height;
    int h2 = info.output_height;
    int c1 = shortcut_tensor->dimension;
    int c2 = info.output_dimension;
    
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;
    
    float *out = cal->weight;
    float *add = shortcut_tensor->weight;

    int i, j, k;
    for(k = 0; k < minc; ++k){
        for(j = 0; j < minh; ++j){
            for(i = 0; i < minw; ++i){
                int out_index = (i * sample + w2 * (j * sample)) * c2 + k;
                int add_index = (i * stride + w1 * (j * stride)) * c1 + k;
                out[out_index] += add[add_index];
            }
        }
    }
    
    delete output_tensor;
    output_tensor = cal;
    return output_tensor;
}

void ShortCutLayer::Backward() {
    float *pre_delta_weight = input_tensor->delta_weight;
    float *grad = output_tensor->delta_weight;
    for (int i = info.input_number; i--; ) {
        *(pre_delta_weight++) += *(grad++);
    }
    
    int w1 = info.output_width;
    int w2 = shortcut_tensor->width;
    int h1 = info.output_height;
    int h2 = shortcut_tensor->height;
    int c1 = info.output_dimension;
    int c2 = shortcut_tensor->dimension;
    
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;
    
    float *out = shortcut_tensor->delta_weight;
    float *add = output_tensor->delta_weight;

    int i, j, k;
    for(k = 0; k < minc; ++k){
        for(j = 0; j < minh; ++j){
            for(i = 0; i < minw; ++i){
                int out_index = (i * sample + w2 * (j * sample)) * c2 + k;
                int add_index = (i * stride + w1 * (j * stride)) * c1 + k;
                out[out_index] += add[add_index];
            }
        }
    }
}
