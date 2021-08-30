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
        case LayerType::LRelu: delete lrelu_layer; break;
        case LayerType::Sigmoid: delete sigmoid_layer; break;
        case LayerType::BatchNormalization: delete batchnorm_layer; break;
        case LayerType::UpSample: delete upsample_layer; break;
        case LayerType::Concat: delete concat_layer; break;
        case LayerType::YOLOv3: delete yolov3_layer; break;
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
    lrelu_layer = nullptr;
    sigmoid_layer = nullptr;
    batchnorm_layer = nullptr;
    upsample_layer = nullptr;
    concat_layer = nullptr;
    yolov3_layer = nullptr;
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
    lrelu_layer = nullptr;
    sigmoid_layer = nullptr;
    batchnorm_layer = nullptr;
    upsample_layer = nullptr;
    concat_layer = nullptr;
    yolov3_layer = nullptr;
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
    lrelu_layer = nullptr;
    sigmoid_layer = nullptr;
    batchnorm_layer = nullptr;
    upsample_layer = nullptr;
    concat_layer = nullptr;
    yolov3_layer = nullptr;
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
            case LayerType::LRelu:
                lrelu_layer = new LReluLayer(*L.lrelu_layer);
                break;
            case LayerType::Sigmoid:
                sigmoid_layer = new SigmoidLayer(*L.sigmoid_layer);
                break;
            case LayerType::BatchNormalization:
                batchnorm_layer = new BatchNormalizationlayer(*L.batchnorm_layer);
                break;
            case LayerType::UpSample:
                upsample_layer = new UpSampleLayer(*L.upsample_layer);
                break;
            case LayerType::Concat:
                concat_layer = new ConcatLayer(*L.concat_layer);
                break;
            case LayerType::YOLOv3:
                yolov3_layer = new YOLOv3Layer(*L.yolov3_layer);
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
    lrelu_layer = L.lrelu_layer;
    sigmoid_layer = L.sigmoid_layer;
    batchnorm_layer = L.batchnorm_layer;
    upsample_layer = L.upsample_layer;
    concat_layer = L.concat_layer;
    yolov3_layer = L.yolov3_layer;
    L.input_layer = nullptr;
    L.fullyconnected_layer = nullptr;
    L.relu_layer = nullptr;
    L.prelu_layer = nullptr;
    L.softmax_layer = nullptr;
    L.convolution_layer = nullptr;
    L.pooling_layer = nullptr;
    L.euclideanloss_layer = nullptr;
    L.shortcut_layer = nullptr;
    L.lrelu_layer = nullptr;
    L.sigmoid_layer = nullptr;
    L.batchnorm_layer = nullptr;
    L.upsample_layer = nullptr;
    L.concat_layer = nullptr;
    L.yolov3_layer = nullptr;
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
    lrelu_layer = nullptr;
    sigmoid_layer = nullptr;
    batchnorm_layer = nullptr;
    upsample_layer = nullptr;
    concat_layer = nullptr;
    yolov3_layer = nullptr;
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
            case LayerType::LRelu:
                lrelu_layer = new LReluLayer(*L.lrelu_layer);
                break;
            case LayerType::Sigmoid:
                sigmoid_layer = new SigmoidLayer(*L.sigmoid_layer);
                break;
            case LayerType::BatchNormalization:
                batchnorm_layer = new BatchNormalizationlayer(*L.batchnorm_layer);
                break;
            case LayerType::UpSample:
                upsample_layer = new UpSampleLayer(*L.upsample_layer);
                break;
            case LayerType::Concat:
                concat_layer = new ConcatLayer(*L.concat_layer);
                break;
            case LayerType::YOLOv3:
                yolov3_layer = new YOLOv3Layer(*L.yolov3_layer);
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
    lrelu_layer = nullptr;
    sigmoid_layer = nullptr;
    batchnorm_layer = nullptr;
    upsample_layer = nullptr;
    concat_layer = nullptr;
    yolov3_layer = nullptr;
    type = string_to_type(opt_["type"]);
    
    switch (type) {
        case LayerType::Input:
            input_layer = new InputLayer(opt_); break;
        case LayerType::Fullyconnected:
            fullyconnected_layer = new FullyConnectedLayer(opt_); break;
        case LayerType::Relu:
            relu_layer = new ReluLayer(opt_); break;
        case LayerType::Softmax:
            softmax_layer = new SoftmaxLayer(opt_); break;
        case LayerType::Convolution:
            convolution_layer = new ConvolutionLayer(opt_); break;
        case LayerType::Pooling:
            pooling_layer = new PoolingLayer(opt_); break;
        case LayerType::EuclideanLoss:
            euclideanloss_layer = new EuclideanLossLayer(opt_); break;
        case LayerType::PRelu:
            prelu_layer = new PReluLayer(opt_); break;
        case LayerType::ShortCut:
            shortcut_layer = new ShortCutLayer(opt_); break;
        case LayerType::LRelu:
            lrelu_layer = new LReluLayer(opt_); break;
        case LayerType::Sigmoid:
            sigmoid_layer = new SigmoidLayer(opt_); break;
        case LayerType::BatchNormalization:
            batchnorm_layer = new BatchNormalizationlayer(opt_); break;
        case LayerType::UpSample:
            upsample_layer = new UpSampleLayer(opt_); break;
        case LayerType::Concat:
            concat_layer = new ConcatLayer(opt_); break;
        case LayerType::YOLOv3:
            yolov3_layer = new YOLOv3Layer(opt_); break;
        default:
            fprintf(stderr, "Unknown layer type!\n"); break;
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
    } else if (type == "LRelu") {
        return LayerType::LRelu;
    } else if (type == "Sigmoid") {
        return LayerType::Sigmoid;
    } else if (type == "BatchNormalization") {
        return LayerType::BatchNormalization;
    } else if (type == "UpSample") {
        return LayerType::UpSample;
    } else if (type == "Concat") {
        return LayerType::Concat;
    } else if (type == "YOLOv3") {
        return LayerType::YOLOv3;
    }
    return LayerType::Error;
}

Tensor* Model_Layer::Forward(Tensor* input_tensor_, Tensor* shortcut_tensor_, Forward_Args *args) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->Forward(input_tensor_, args);
        case LayerType::Pooling:
            return pooling_layer->Forward(input_tensor_);
        case LayerType::PRelu:
            return prelu_layer->Forward(input_tensor_);
        case LayerType::Fullyconnected:
            return fullyconnected_layer->Forward(input_tensor_);
        case LayerType::Relu:
            return relu_layer->Forward(input_tensor_);
        case LayerType::ShortCut:
            return shortcut_layer->Forward(input_tensor_, shortcut_tensor_);
        case LayerType::Softmax:
            return softmax_layer->Forward(input_tensor_);
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Forward(input_tensor_);
        case LayerType::Input:
            return input_layer->Forward(input_tensor_);
        case LayerType::LRelu:
            return lrelu_layer->Forward(input_tensor_);
        case LayerType::Sigmoid:
            return sigmoid_layer->Forward(input_tensor_);
        case LayerType::BatchNormalization:
            return batchnorm_layer->Forward(input_tensor_, args);
        case LayerType::UpSample:
            return upsample_layer->Forward(input_tensor_);
        case LayerType::Concat:
            return concat_layer->Forward(input_tensor_, shortcut_tensor_);
        case LayerType::YOLOv3:
            return yolov3_layer->Forward(input_tensor_);
        default:
            break;
    }
    return nullptr;
}

float Model_Layer::Backward(vfloat& target) {
    switch (type) {
        case LayerType::Softmax:
            return softmax_layer->Backward(target);
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Backward(target);
        case LayerType::YOLOv3:
            return yolov3_layer->Backward(target);
        default:
            break;
    }
    return 0;
}

void Model_Layer::Backward() {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->Backward();
        case LayerType::Pooling:
            return pooling_layer->Backward();
        case LayerType::PRelu:
            return prelu_layer->Backward();
        case LayerType::Input:
            return input_layer->Backward();
        case LayerType::Fullyconnected:
            return fullyconnected_layer->Backward();
        case LayerType::Relu:
            return relu_layer->Backward();
        case LayerType::ShortCut:
            return shortcut_layer->Backward();
        case LayerType::LRelu:
            return lrelu_layer->Backward();
        case LayerType::Sigmoid:
            return sigmoid_layer->Backward();
        case LayerType::BatchNormalization:
            return batchnorm_layer->Backward();
        case LayerType::UpSample:
            return upsample_layer->Backward();
        case LayerType::Concat:
            return concat_layer->Backward();
        default:
            break;
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
    } else if (type == LayerType::LRelu) {
        lrelu_layer->shape();
    } else if (type == LayerType::Sigmoid) {
        sigmoid_layer->shape();
    } else if (type == LayerType::BatchNormalization) {
        batchnorm_layer->shape();
    } else if (type == LayerType::UpSample) {
        upsample_layer->shape();
    } else if (type == LayerType::Concat) {
        concat_layer->shape();
    } else if (type == LayerType::YOLOv3) {
        yolov3_layer->shape();
    }
}

int Model_Layer::getParameter(int type_) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->getParameter(type_);
        case LayerType::Pooling:
            return pooling_layer->getParameter(type_);
        case LayerType::PRelu:
            return prelu_layer->getParameter(type_);
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getParameter(type_);
        case LayerType::Relu:
            return relu_layer->getParameter(type_);
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->getParameter(type_);
        case LayerType::Softmax:
            return softmax_layer->getParameter(type_);
        case LayerType::Input:
            return input_layer->getParameter(type_);
        case LayerType::ShortCut:
            return shortcut_layer->getParameter(type_);
        case LayerType::LRelu:
            return lrelu_layer->getParameter(type_);
        case LayerType::Sigmoid:
            return sigmoid_layer->getParameter(type_);
        case LayerType::BatchNormalization:
            return batchnorm_layer->getParameter(type_);
        case LayerType::UpSample:
            return upsample_layer->getParameter(type_);
        case LayerType::Concat:
            return concat_layer->getParameter(type_);
        case LayerType::YOLOv3:
            return yolov3_layer->getParameter(type_);
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
    } else if (type == LayerType::LRelu) {
        return lrelu_layer->save(f);
    } else if (type == LayerType::Sigmoid) {
        return sigmoid_layer->save(f);
    } else if (type == LayerType::BatchNormalization) {
        return batchnorm_layer->save(f);
    } else if (type == LayerType::UpSample) {
        return upsample_layer->save(f);
    } else if (type == LayerType::Concat) {
        return concat_layer->save(f);
    } else if (type == LayerType::YOLOv3) {
        return yolov3_layer->save(f);
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
    } else if (type == LayerType::LRelu) {
        return lrelu_layer->load(f);
    } else if (type == LayerType::Sigmoid) {
        return sigmoid_layer->load(f);
    } else if (type == LayerType::BatchNormalization) {
        return batchnorm_layer->load(f);
    } else if (type == LayerType::UpSample) {
        return upsample_layer->load(f);
    } else if (type == LayerType::Concat) {
        return concat_layer->load(f);
    } else if (type == LayerType::YOLOv3) {
        return yolov3_layer->load(f);
    }
    return 0;
}

void Model_Layer::ClearGrad() {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->ClearGrad();
        case LayerType::Pooling:
            return pooling_layer->ClearGrad();
        case LayerType::PRelu:
            return prelu_layer->ClearGrad();
        case LayerType::Fullyconnected:
            return fullyconnected_layer->ClearGrad();
        case LayerType::Relu:
            return relu_layer->ClearGrad();
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->ClearGrad();
        case LayerType::Softmax:
            return softmax_layer->ClearGrad();
        case LayerType::Input:
            return input_layer->ClearGrad();
        case LayerType::ShortCut:
            return shortcut_layer->ClearGrad();
        case LayerType::LRelu:
            return lrelu_layer->ClearGrad();
        case LayerType::Sigmoid:
            return sigmoid_layer->ClearGrad();
        case LayerType::BatchNormalization:
            return batchnorm_layer->ClearGrad();
        case LayerType::UpSample:
            return upsample_layer->ClearGrad();
        case LayerType::Concat:
            return concat_layer->ClearGrad();
        case LayerType::YOLOv3:
            return yolov3_layer->ClearGrad();
        default:
            break;
    }
}

Train_Args Model_Layer::getTrainArgs() {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->getTrainArgs();
        case LayerType::PRelu:
            return prelu_layer->getTrainArgs();
        case LayerType::Fullyconnected:
            return fullyconnected_layer->getTrainArgs();
        case LayerType::BatchNormalization:
            return batchnorm_layer->getTrainArgs();
        default:
            return Train_Args();
    }
    return Train_Args();
}

int Model_Layer::getWorkspaceSize() {
    switch(type) {
        case LayerType::Convolution:
            return convolution_layer->info.workspace_size;
        default:
            return 0;
    }
    return 0;
}

Train_Args BaseLayer::getTrainArgs() {
    switch(type) {
        case LayerType::Convolution:
            return Train_Args(kernel, biases, info.kernel_width * info.kernel_height * info.input_dimension, info.output_dimension, (info.batchnorm) ? 0 : info.output_dimension, vfloat{0, 1});
        case LayerType::Fullyconnected:
            return Train_Args(kernel, biases, info.input_number, info.output_dimension, info.output_dimension, vfloat{0, 1});
        case LayerType::PRelu:
            return Train_Args(kernel, kernel, info.input_number, 1, 0, vfloat{0, 0});
        case LayerType::BatchNormalization:
            return Train_Args(kernel, biases, info.output_dimension, 3, info.output_dimension, vfloat{0, 0});
        default:
            return Train_Args();
    }
    return Train_Args();
}

BaseLayer::~BaseLayer() {
    if (output_tensor)
        delete output_tensor;
    if (kernel)
        delete [] kernel;
    if (biases)
        delete biases;
    if (output_index)
        delete [] output_index;
    if (input_index)
        delete [] input_index;
}

BaseLayer::BaseLayer() {
    input_tensor = nullptr;
    output_tensor = nullptr;
    kernel = nullptr;
    biases = nullptr;
    output_index = nullptr;
    input_index = nullptr;
}

BaseLayer::BaseLayer(BaseLayer *L) {
    if (this != L) {
        type = L->type;
        name = L->name;
        input_name = L->input_name;
        info = L->info;
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
            kernel = new Tensor [info.kernel_num];
            for (int i = 0; i < info.kernel_num; ++i) {
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
        if (L->output_index) {
            output_index = new int [info.output_index_size];
            memcpy(output_index, L->output_index, sizeof(int) * info.output_index_size);
        } else {
            output_index = nullptr;
        }
        if (L->input_index) {
            input_index = new int [info.input_index_size];
            memcpy(input_index, L->input_index, sizeof(int) * info.input_index_size);
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
            kernel = new Tensor [info.kernel_num];
            for (int i = 0; i < info.kernel_num; ++i) {
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
        if (L.output_index) {
            output_index = new int [info.output_index_size];
            memcpy(output_index, L.output_index, sizeof(int) * info.output_index_size);
        } else {
            output_index = nullptr;
        }
        if (L.input_index) {
            input_index = new int [info.input_index_size];
            memcpy(input_index, L.input_index, sizeof(int) * info.input_index_size);
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
    opt = L.opt;
    input_tensor = L.input_tensor;
    L.input_tensor = nullptr;
    output_tensor = L.output_tensor;
    L.output_tensor = nullptr;
    kernel = L.kernel;
    L.kernel = nullptr;
    biases = L.biases;
    L.biases = nullptr;
    output_index = L.output_index;
    L.output_index = nullptr;
    input_index = L.input_index;
    L.input_index = nullptr;
}

BaseLayer& BaseLayer::operator=(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
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
            kernel = new Tensor [info.kernel_num];
            for (int i = 0; i < info.kernel_num; ++i) {
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
        if (L.output_index) {
            output_index = new int [info.output_index_size];
            memcpy(output_index, L.output_index, sizeof(int) * info.output_index_size);
        } else {
            output_index = nullptr;
        }
        if (L.input_index) {
            input_index = new int [info.input_index_size];
            memcpy(input_index, L.input_index, sizeof(int) * info.input_index_size);
        } else {
            input_index = nullptr;
        }
    }
    return *this;
}

string BaseLayer::type_to_string() {
    switch(type) {
        case Input: return "Input";
        case Fullyconnected: return "Fullyconnected";
        case Relu: return "Relu";
        case PRelu: return "PRelu";
        case Softmax: return "Softmax";
        case Convolution: return "Convolution";
        case Pooling: return "Pooling";
        case EuclideanLoss: return "EuclideanLoss";
        case ShortCut: return "ShortCut";
        case LRelu: return "LRelu";
        case Sigmoid: return "Sigmoid";
        case BatchNormalization: return "BatchNorm";
        case UpSample: return "UpSample";
        case Concat: return "Concat";
        case YOLOv3: return "YOLOv3";
        case Error: return "Error";
    }
    return "Unknown";
}

void BaseLayer::shape() {
    printf("%-17s%-13s %-13s  ", type_to_string().c_str(), name.c_str(), input_name.c_str());
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
//    } else if (type == LayerType::BatchNormalization) {
//        printf("Scale:\n");
//        kernel[0].showWeight();
//        printf("Running mean:\n");
//        kernel[1].showWeight();
//        printf("Running variance:\n");
//        kernel[2].showWeight();
//        printf("Bias:\n");
//        biases->showWeight();
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
        fwrite(&info.kernel_width, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.kernel_height, sizeof(int), 1, f);
        fwrite(&info.stride, sizeof(int), 1, f);
        fwrite(&info.padding, sizeof(int), 1, f);
        for (int i = 0; i < info.output_dimension; ++i) {
            kernel[i].save(f);
        }
        biases->save(f);
    } else if (type == LayerType::Relu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Pooling) {
        fwrite(&info.kernel_width, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.kernel_height, sizeof(int), 1, f);
        fwrite(&info.stride, sizeof(int), 1, f);
        fwrite(&info.padding, sizeof(int), 1, f);
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
    } else if (type == LayerType::LRelu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Sigmoid) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::BatchNormalization) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        kernel[0].save(f);
        kernel[3].save(f);
        kernel[4].save(f);
        biases->save(f);
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
    } else if (type == LayerType::BatchNormalization) {
        kernel[0].load(f);
        kernel[3].load(f);
        kernel[4].load(f);
        biases->load(f);
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
    
    info.kernel_width = atoi(opt["kernel_width"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.kernel_height = (opt.find("kernel_height") == opt.end()) ? info.kernel_width : atoi(opt["kernel_height"].c_str());
    info.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info.kernel_width - 1) / 2) : atoi(opt["padding"].c_str()));
    
    info.output_width = (info.input_width + info.padding * 2 - info.kernel_width) / info.stride + 1;
    info.output_height = (info.input_height + info.padding * 2 - info.kernel_height) / info.stride + 1;
    info.output_dimension = atoi(opt["number_kernel"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    info.batchnorm = (opt.find("batchnorm") != opt.end());
    
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(info.kernel_width, info.kernel_height, info.input_dimension);
        kernel[i] = new_kernel;
    }
    info.kernel_num = info.output_dimension;
    
    biases = new Tensor(1, 1, info.output_dimension, bias);
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int stride = info.stride;
    int neg_padding = -info.padding;
    int out_height, out_width;
    int x, y;
    int kernel_w, kernel_h, kernel_dim;
    int output_height = info.output_height;
    int output_width = info.output_width;
    int coordinate_x, coordinate_y;
    int kernel_width = info.kernel_width;
    int kernel_height = info.kernel_height;
    int input_dimension = info.input_dimension;
    
    int kernel_n = kernel_width * kernel_height * input_dimension;
    int out_size = output_width * output_height;
    int workspace_size = out_size * kernel_n;
    int count;
    
    info.workspace_size = workspace_size;
    info.input_index_size = workspace_size;
    
    input_index = new int [workspace_size];
    fill(input_index, input_index + workspace_size, -1);
    
    count = -1;
    y = neg_padding;
    for (out_height = 0; out_height < output_height; y += stride, ++out_height) {
        x = neg_padding;
        for (out_width = 0; out_width < output_width; x += stride, ++out_width)
        for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
            coordinate_y = y + kernel_h;
            for (kernel_dim = 0; kernel_dim < input_dimension; ++kernel_dim) {
                for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                    coordinate_x = x + kernel_w;
                    if (coordinate_y >= 0 && coordinate_y < input_height && coordinate_x >= 0 && coordinate_x < input_width)
                        input_index[++count] = ((input_width * coordinate_y) + coordinate_x) + (input_width * input_height) * kernel_dim;
                    else
                        count++;
                }
            }
        }
    }
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* ConvolutionLayer::Forward(Tensor *input_tensor_, Forward_Args *args) {
    input_tensor = input_tensor_;
    
    int kernel_n = info.kernel_width * info.kernel_height * info.input_dimension;
    int out_size = info.output_width * info.output_height;
    
    int one_batch_input_size = info.input_number;
    
    float *input_weight = input_tensor->weight;
    float *output_weight = output_tensor->weight;
    float *workspace = args->workspace;
    
    for (int b = info.batch_size; b--; ) {
        float *bias_ptr = biases->weight;
        
        int count;
        int index;
        for (count = info.workspace_size; count--; ) {
            workspace[count] = ((index = input_index[count]) == -1) ? 0 : input_weight[index];
        }
        
        float conv, bias;
        float *kernel_weight, *kernel_weight_ptr;
        
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
                *(output_weight++) = conv + bias;
            }
        }
        input_weight += one_batch_input_size;
    }
    return output_tensor;
}

void ConvolutionLayer::Backward() {
    float *kernel_weight, *kernel_grad;
    int kernel_n = info.kernel_width * info.kernel_height * info.input_dimension;
    
    int *input_index_ptr;
    
    float chain_grad;
    int out_size = info.output_width * info.output_height;
    
    int index;
    int out_index;
    int kernel_index;
    
    int one_batch_input_size = info.input_number;
    
    float *input_weight = input_tensor->weight;
    float *input_grad = input_tensor->delta_weight;
    float *output_grad = output_tensor->delta_weight;
    
    for (int b = info.batch_size; b--; ) {
        float *bias_grad = biases->delta_weight;
        
        for (int out_dim = 0; out_dim < info.output_dimension; ++out_dim) {
            Tensor *kernel_act = kernel + out_dim;
            kernel_weight = kernel_act->weight;
            kernel_grad = kernel_act->delta_weight;
            input_index_ptr = input_index;
            
            for (out_index = out_size; out_index; --out_index) {
                chain_grad = *(output_grad++);
                for (kernel_index = 0; kernel_index < kernel_n; ++kernel_index) {
                    if ((index = *(input_index_ptr++)) != -1) {
                        kernel_grad[kernel_index] += input_weight[index] * chain_grad;
                        input_grad[index] += kernel_weight[kernel_index] * chain_grad;
                    }
                }
                bias_grad[out_dim] += chain_grad;
            }
        }
        input_weight += one_batch_input_size;
        input_grad += one_batch_input_size;
    }
}

PoolingLayer::PoolingLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Pooling;
    name = (opt.find("name") == opt.end()) ? "pool" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.kernel_width = atoi(opt["kernel_width"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.kernel_height = (opt.find("kernel_height") == opt.end()) ? info.kernel_width : atoi(opt["kernel_height"].c_str());
    info.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info.kernel_width - 1)) : atoi(opt["padding"].c_str()));
    
    info.output_dimension = info.input_dimension;
    info.output_width = (info.input_width + info.padding - info.kernel_width) / info.stride + 1;
    info.output_height = (info.input_height + info.padding - info.kernel_height) / info.stride + 1;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    choosex.assign(info.output_number * info.batch_size, 0);
    choosey.assign(info.output_number * info.batch_size, 0);
    
    int output_dimension = info.output_dimension;
    int output_width = info.output_width;
    int output_height = info.output_height;
    int kernel_width = info.kernel_width;
    int kernel_height = info.kernel_height;
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_channel_size = input_width * input_height;
    int stride = info.stride;
    int neg_padding = -info.padding / 2;
    
    input_index = new int [info.output_number * kernel_width * kernel_height];
    fill(input_index, input_index + info.output_number * kernel_width * kernel_height, -1);
    
    int input_counter = 0;
    
    info.input_index_size = info.output_number * kernel_width * kernel_height;
    
    for (int output_d = 0; output_d < output_dimension; ++output_d) {
        int offset_h = neg_padding;
        for (int output_h = 0; output_h < output_height; ++output_h, offset_h += stride) {
            int offset_w = neg_padding;
            for (int output_w = 0; output_w < output_width; ++output_w, offset_w += stride) {
                for (int kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                    int act_h = offset_h + kernel_h;
                    for (int kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
                        int act_w = offset_w + kernel_w;
                        if (act_w >= 0 && act_w < input_width && act_h >= 0 && act_h < input_height) {
                            input_index[input_counter] = ((act_h * input_width) + act_w) + (input_channel_size) * output_d;
                        }
                        ++input_counter;
                    }
                }
            }
        }
    }
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
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
    int neg_padding = -info.padding / 2;
    int stride = info.stride;
    int kernel_width = info.kernel_width;
    int kernel_height = info.kernel_height;
    
    int counter = 0;
    
    int one_batch_input_size = info.input_number;
    
    float *input_weight = input_tensor->weight;
    float *output_weight = output_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        int *input_index_ptr = input_index;
        int index;
        for (int output_d = 0; output_d < output_dimension; ++output_d) {
            offset_h = neg_padding;
            for (output_h = 0; output_h < output_height; ++output_h, offset_h += stride) {
                offset_w = neg_padding;
                for (output_w = 0; output_w < output_width; ++output_w, offset_w += stride) {
                    maximum = -100000000;
                    for (kernel_h = 0; kernel_h < kernel_height; ++kernel_h) {
                        for (kernel_w = 0; kernel_w < kernel_width; ++kernel_w) {
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
                    *(output_weight++) = maximum;
                }
            }
        }
        input_weight += one_batch_input_size;
    }
    return output_tensor;
}

void PoolingLayer::Backward() {
    Tensor *input = input_tensor;

    int counter = 0;
    float chain_grad;
    int out_size = info.output_width * info.output_height;
    
    float *output_grad = output_tensor->delta_weight;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int cal_d = 0; cal_d < info.output_dimension; ++cal_d) {
            for (int out_counter = 0; out_counter < out_size; ++out_counter) {
                chain_grad = *(output_grad++);
                input->addGrad(choosex[counter], choosey[counter], cal_d, chain_grad, b * info.input_number);
                ++counter;
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
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    kernel = new Tensor [info.output_dimension];
    for (int i = 0; i < info.output_dimension; ++i) {
        Tensor new_kernel(1, 1, info.input_number);
        kernel[i] = new_kernel;
    }
    info.kernel_num = info.output_dimension;
    
    float bias = atof(opt["bias"].c_str());
    biases = new Tensor(1, 1, info.output_dimension, bias);
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
}

Tensor* FullyConnectedLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *input = input_tensor_->weight;
    float *pos = output_tensor->weight;
    
    float sum;
    float *weight;
    
    int i, j;
    float *input_ptr;
    
    for (int b = info.batch_size; b--; ) {
        float *bias = biases->weight;
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
        input += one_batch_input_size;
        pos += one_batch_output_size;
    }
    return output_tensor;
}

void FullyConnectedLayer::Backward() {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *cal_w = input_tensor->weight;
    float *cal_dw = input_tensor->delta_weight;
    float *output_grad = output_tensor->delta_weight;
    
    float *act_weight, *act_grad;
    float chain_grad;
    
    for (int b = info.batch_size; b--; ) {
        float *act_biases_grad = biases->delta_weight;
        
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
        cal_w += one_batch_input_size;
        cal_dw += one_batch_input_size;
        output_grad += one_batch_output_size;
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
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* ReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    float value;
    
    for (int i = info.input_number * info.batch_size; i--; ) {
        value = *(val++);
        *(out++) = (value < 0) ? 0 : value;
    }
    return output_tensor;
}

void ReluLayer::Backward() {
    int one_batch_size = info.input_number;
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = input_tensor->delta_weight;
    
    for (int b = info.batch_size; b--; ) {
        for (int i = info.input_number; i--; ) {
            if (act_weight[i] <= 0)
                pos_grad[i] += 0;
            else
                pos_grad[i] += act_grad[i];
        }
        act_weight += one_batch_size;
        act_grad += one_batch_size;
        pos_grad += one_batch_size;
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
    info.batch_size = atoi(opt["batch_size"].c_str());
    float alpha = (opt.find("alpha") == opt.end()) ? 0.25 : atof(opt["alpha"].c_str());
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension, alpha);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* PReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        float *alpha = kernel->weight;
        float value;
        for (int i = info.input_number; i--; ++alpha) {
            value = *(val++);
            *(out++) = (value < 0) ? value * *alpha : value;
        }
    }
    
    return output_tensor;
}

void PReluLayer::Backward() {
    int one_batch_size = info.input_number;
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = input_tensor->delta_weight;
    
    for (int b = info.batch_size; b--; ) {
        float *alpha = kernel->weight;
        float *alpha_grad = kernel->delta_weight;
        float chain_grad;
        
        for (int i = 0; i < info.input_number; ++i) {
            chain_grad = act_grad[i];
            if (act_weight[i] < 0) {
                pos_grad[i] += chain_grad * alpha[i];
                alpha_grad[i] += act_weight[i] * chain_grad;
            }
            else {
                pos_grad[i] += chain_grad;
                alpha_grad[i] += 0;
            }
        }
        act_weight += one_batch_size;
        act_grad += one_batch_size;
        pos_grad += one_batch_size;
    }
}

SoftmaxLayer::SoftmaxLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Softmax;
    name = (opt.find("name") == opt.end()) ? "sm" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = 1;
    info.output_height = 1;
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str()) * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.output_dimension * info.batch_size, 0);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
}

Tensor* SoftmaxLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *act = input_tensor_->weight;
    float *expo_sum_ptr = kernel->weight;
    float *cal_weight = output_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        float max = act[0];
        for (int i = 1; i < info.output_dimension; ++i) {
            if (act[i] > max)
                max = act[i];
        }
        float sum = 0;
        for (int i = 0; i < info.output_dimension; ++i) {
            sum += (expo_sum_ptr[i] = exp(act[i] - max));
        }
        for (int i = 0; i < info.output_dimension; ++i) {
            expo_sum_ptr[i] /= sum;
            cal_weight[i] = expo_sum_ptr[i];
        }
        act += one_batch_input_size;
        expo_sum_ptr += one_batch_output_size;
        cal_weight += one_batch_output_size;
    }
    return output_tensor;
}

float SoftmaxLayer::Backward(vfloat& target) {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *cal_delta_weight = input_tensor->delta_weight;
    float *expo_sum_ptr = kernel->weight;
    
    int output_dimension = info.output_dimension;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < output_dimension; ++i) {
            float indicator = (i == target[b]) ? 1.0 : 0.0;
            float mul = -(indicator - expo_sum_ptr[i]);
            cal_delta_weight[i] = mul;
        }
        cal_delta_weight += one_batch_input_size;
        expo_sum_ptr += one_batch_output_size;
    }
    float loss = 0;
    expo_sum_ptr = kernel->weight;
    for (int b = 0; b < info.batch_size; ++b) {
        loss += -log(expo_sum_ptr[(int)target[b]]);
        expo_sum_ptr += one_batch_output_size;
    }
    return loss;
}

EuclideanLossLayer::EuclideanLossLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::EuclideanLoss;
    name = (opt.find("name") == opt.end()) ? "el" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_dimension = atoi(opt["input_dimension"].c_str()) * atoi(opt["input_width"].c_str()) * atoi(opt["input_height"].c_str());
    info.output_width = 1;
    info.output_height = 1;
    info.batch_size = atoi(opt["batch_size"].c_str());
}

Tensor* EuclideanLossLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    output_tensor = input_tensor;
    return output_tensor;
}

float EuclideanLossLayer::Backward(vfloat& target) {
    Tensor *cal_tensor = input_tensor;
    
    int one_batch_size = info.output_dimension;
    
    float *cal_weight = cal_tensor->weight;
    float *cal_delta_weight = cal_tensor->delta_weight;
    float loss = 0;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < info.output_dimension; ++i) {
            float delta = cal_weight[i] - target[i + b * one_batch_size];
            cal_delta_weight[i] += delta;
            loss += 0.5 * delta * delta;
        }
        cal_weight += one_batch_size;
        cal_delta_weight += one_batch_size;
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
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* ShortCutLayer::Forward(Tensor *input_tensor_, Tensor *shortcut_tensor_) {
    input_tensor = input_tensor_;
    shortcut_tensor = shortcut_tensor_;
    
    int w1 = shortcut_tensor->width;
    int w2 = info.output_width;
    int h1 = shortcut_tensor->height;
    int h2 = info.output_height;
    int c1 = shortcut_tensor->dimension / info.batch_size;
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
    
    int one_batch_input_size = info.input_number;
    int output_channel_size = w2 * h2;
    int one_batch_output_size = info.input_number;
    int shortcut_channel_size = w1 * h1;
    int one_batch_shortcut_size = shortcut_tensor->size / info.batch_size;
    
    float *in = input_tensor->weight;
    float *out = output_tensor->weight;
    float *add = shortcut_tensor->weight;
    
    for (int b = info.batch_size; b--; ) {
        int i, j, k;
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = (i * sample + w2 * (j * sample)) + output_channel_size * k;
                    int add_index = (i * stride + w1 * (j * stride)) + shortcut_channel_size * k;
                    out[out_index] = in[out_index] + add[add_index];
                }
            }
        }
        in += one_batch_input_size;
        out += one_batch_output_size;
        add += one_batch_shortcut_size;
    }
    
    return output_tensor;
}

void ShortCutLayer::Backward() {
    float *pre_delta_weight = input_tensor->delta_weight;
    float *grad = output_tensor->delta_weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(pre_delta_weight++) += *(grad++);
    }
    
    int w1 = info.output_width;
    int w2 = shortcut_tensor->width;
    int h1 = info.output_height;
    int h2 = shortcut_tensor->height;
    int c1 = info.output_dimension;
    int c2 = shortcut_tensor->dimension / info.batch_size;
    
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;
    
    int one_batch_output_size = info.input_number;
    int one_batch_shortcut_size = shortcut_tensor->size / info.batch_size;
    
    float *out = shortcut_tensor->delta_weight;
    int shortcut_channel_size = w2 * h2;
    float *add = output_tensor->delta_weight;
    int output_channel_size = w1 * h1;
    
    for (int b = info.batch_size; b--; ) {
        int i, j, k;
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = (i * sample + w2 * (j * sample)) + shortcut_channel_size * k;
                    int add_index = (i * stride + w1 * (j * stride)) + output_channel_size * k;
                    out[out_index] += add[add_index];
                }
            }
        }
        out += one_batch_shortcut_size;
        add += one_batch_output_size;
    }
}

LReluLayer::LReluLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::LRelu;
    name = (opt.find("name") == opt.end()) ? "lrelu" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* LReluLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    float value;
    for (int i = info.input_number * info.batch_size; i--; ++val) {
        value = *val;
        *(out++) = (value < 0) ? value * 0.01 : value;
    }
    return output_tensor;
}

void LReluLayer::Backward() {
    int one_batch_size = info.input_number;
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = input_tensor->delta_weight;
    float chain_grad;
    
    for (int b = info.batch_size; b--; ) {
        for (int i = 0; i < info.input_number; ++i) {
            chain_grad = act_grad[i];
            if (act_weight[i] < 0) {
                pos_grad[i] += chain_grad * 0.01;
            }
            else {
                pos_grad[i] += chain_grad;
            }
        }
        act_weight += one_batch_size;
        act_grad += one_batch_size;
        pos_grad += one_batch_size;
    }
}

SigmoidLayer::SigmoidLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Sigmoid;
    name = (opt.find("name") == opt.end()) ? "sig" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* SigmoidLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(out++) = 1.0 / (1.0 + exp(-(*(val++))));
    }
    return output_tensor;
}

void SigmoidLayer::Backward() {
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = input_tensor->delta_weight;
    float value;
    
    for (int i = 0; i < info.input_number * info.batch_size; ++i) {
        value = *(act_weight++);
        *(pos_grad++) = value * (1.0 - value) * *(act_grad++);
    }
}

BatchNormalizationlayer::BatchNormalizationlayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::BatchNormalization;
    name = (opt.find("name") == opt.end()) ? "bn" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    kernel = new Tensor [7];
    kernel[0] = Tensor(1, 1, info.output_dimension, 1); // scale
    kernel[1] = Tensor(1, 1, info.output_dimension, 0); // mean
    kernel[2] = Tensor(1, 1, info.output_dimension, 0); // variance
    kernel[3] = Tensor(1, 1, info.output_dimension, 0); // running mean
    kernel[4] = Tensor(1, 1, info.output_dimension, 0); // running variance
    kernel[5] = Tensor(1, 1, info.input_number * info.batch_size, 0); // x
    kernel[6] = Tensor(1, 1, info.input_number * info.batch_size, 0); // x_norm
    info.kernel_num = 7;
    
    biases = new Tensor(1, 1, info.output_dimension, 0);
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

Tensor* BatchNormalizationlayer::Forward(Tensor *input_tensor_, Forward_Args *args) {
    input_tensor = input_tensor_;
    
    float *src = input_tensor->weight;
    float *output = output_tensor->weight;
    float *bias = biases->weight;
    float *scale = kernel[0].weight;
    float *mean = kernel[1].weight;
    float *variance = kernel[2].weight;
    float *running_mean = kernel[3].weight;
    float *running_variance = kernel[4].weight;
    float *x = kernel[5].weight;
    float *x_norm = kernel[6].weight;
    
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int total_size = info.input_number * batch_size;
    int channel_size = info.output_width * info.output_height;
    
    copy_cpu(total_size, src, output);
    copy_cpu(total_size, output, x);
    
    if (args->train) {
        cal_mean(output, batch_size, output_dimension, channel_size, mean);
        cal_variance(output, mean, batch_size, output_dimension, channel_size, variance);
        scal_cpu(output_dimension, 0.99, running_mean);
        axpy_cpu(output_dimension, 0.01, mean, running_mean);
        scal_cpu(output_dimension, 0.99, running_variance);
        axpy_cpu(output_dimension, 0.01, variance, running_variance);
        
        normalize(output, mean, variance, batch_size, output_dimension, channel_size);
        copy_cpu(total_size, output, x_norm);
    } else {
        normalize(output, running_mean, running_variance, batch_size, output_dimension, channel_size);
    }
    
    float scale_value, bias_value;
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < output_dimension; ++d) {
            scale_value = scale[d];
            bias_value = bias[d];
            for (int i = 0; i < channel_size; ++i) {
                *output = *output * scale_value + bias_value;
                ++output;
            }
        }
    }
    
    return output_tensor;
}

void BatchNormalizationlayer::Backward() {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *bias_delta = biases->delta_weight;
    float *scale_delta = kernel[0].delta_weight;
    float *mean_delta = kernel[1].delta_weight;
    float *variance_delta = kernel[2].delta_weight;
    
    float *scale = kernel[0].weight;
    float *mean = kernel[1].weight;
    float *variance = kernel[2].weight;
    float *x = kernel[5].weight;
    float *x_norm = kernel[6].weight;
    
    int batch_size = info.batch_size;
    int output_dimension = info.output_dimension;
    int one_batch_size = info.input_number;
    int channel_size = info.output_width * info.output_height;
    
    float chain_grad, scale_value, mean_value;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < output_dimension; ++d) {
            float &bias_delta_value = bias_delta[d];
            float &scale_delta_value = scale_delta[d];
            float &mean_delta_value = mean_delta[d];
            float &variance_delta_value = variance_delta[d];
            scale_value = scale[d];
            mean_value = mean[d];
            for (int i = 0; i < channel_size; ++i) {
                chain_grad = *output_delta;
                bias_delta_value += chain_grad;
                scale_delta_value += chain_grad * *(x_norm++);
                chain_grad = *(output_delta++) *= scale_value;
                mean_delta_value += chain_grad;
                variance_delta_value += chain_grad * (*(x++) - mean_value);
            }
        }
    }
    
    for (int d = 0; d < output_dimension; ++d) {
        mean_delta[d] *= (-1.0 / sqrt(variance[d] + .00001f));
        variance_delta[d] *= (-0.5 * pow(variance[d] + 0.00001f, (float)(-1.5)));
    }
    
    output_delta = output_tensor->delta_weight;
    x = kernel[5].weight;
    
    float variance_scale;
    float variance_delta_value;
    float mean_delta_value;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < output_dimension; ++d) {
            mean_value = mean[d];
            mean_delta_value = mean_delta[d] / (channel_size * batch_size);
            variance_scale = 1.0 / (sqrt(variance[d] + .00001f));
            variance_delta_value = variance_delta[d];
            for (int i = 0; i < channel_size; ++i) {
                *output_delta = *output_delta * variance_scale + variance_delta_value * 2.0 * (*(x++) - mean_value) / (channel_size * batch_size) + mean_delta_value;
                ++output_delta;
            }
        }
    }
    
    output_delta = output_tensor->delta_weight;
    copy_cpu(one_batch_size * batch_size, output_delta, input_delta);
}

UpSampleLayer::UpSampleLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::UpSample;
    name = (opt.find("name") == opt.end()) ? "up" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    int stride = atoi(opt["stride"].c_str());
    info.output_width = stride * info.input_width;
    info.output_height = stride * info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.reverse = false;
    
    if (stride < 0) {
        stride = -stride;
        info.reverse = true;
        info.output_width = info.input_width / stride;
        info.output_height = info.input_height / stride;
    }
    info.stride = stride;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

Tensor* UpSampleLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    
    if (info.reverse) {
        
    } else {
        upsample(input_tensor->weight, output_tensor->weight, info.batch_size, info.input_width, info.input_height, info.input_dimension, info.stride, true);
    }
    
    return output_tensor;
}

void UpSampleLayer::Backward() {
    if (info.reverse) {
        
    } else {
        upsample(output_tensor->delta_weight, input_tensor->delta_weight, info.batch_size, info.input_width, info.input_height, info.input_dimension, info.stride, false);
    }
}

void UpSampleLayer::upsample(float *src, float *dst, int batch_size, int width, int height, int dimension, int stride, bool forward) {
    int one_batch_input_size = info.input_number;
    int input_channel_size = width * height;
    int one_batch_output_size = info.output_number;
    int output_channel_size = input_channel_size * stride * stride;
    
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < dimension; ++d) {
            for (int h = 0; h < height * stride; ++h) {
                for (int w = 0; w < width * stride; ++w) {
                    int input_index = b * one_batch_input_size + ((h / stride * width) + w / stride) + input_channel_size * d;
                    int output_index = b * one_batch_output_size + ((h * width) + w) + output_channel_size * d;
                    if (forward)
                        dst[output_index] = src[input_index];
                    else
                        src[input_index] += dst[output_index];
                }
            }
        }
    }
}

ConcatLayer::ConcatLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Concat;
    name = (opt.find("name") == opt.end()) ? "cc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    int concat_width = atoi(opt["concat_width"].c_str());
    int concat_height = atoi(opt["concat_height"].c_str());
    assert(info.output_width == concat_width);
    assert(info.output_height == concat_height);
    info.concat_dimension = atoi(opt["concat_dimension"].c_str());
    info.output_dimension = info.input_dimension + info.concat_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

Tensor* ConcatLayer::Forward(Tensor *input_tensor_, Tensor *concat_tensor_) {
    input_tensor = input_tensor_;
    concat_tensor = concat_tensor_;
    
    float *src = input_tensor->weight;
    float *concat = concat_tensor->weight;
    float *output = output_tensor->weight;
    
    int concat_size = info.input_width * info.input_height * info.concat_dimension;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = info.input_number; i--; )
            *(output++) = *(src++);
        for (int i = concat_size; i--; )
            *(output++) = *(concat++);
    }
    
    return output_tensor;
}

void ConcatLayer::Backward() {
    float *src_delta = input_tensor->delta_weight;
    float *concat_delta = concat_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    
    int concat_size = info.input_width * info.input_height * info.concat_dimension;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = info.input_number; i--; )
            *(src_delta++) += *(output_delta++);
        for (int i = concat_size; i--; )
            *(concat_delta++) += *(output_delta++);
    }
}

YOLOv3Layer::YOLOv3Layer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::YOLOv3;
    name = (opt.find("name") == opt.end()) ? "yolov3" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    
    info.output_width = info.input_width;
    info.output_height = info.input_height;
    info.output_dimension = info.input_dimension;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
    info.total_anchor_num = atoi(opt["total_anchor_num"].c_str());
    info.anchor_num = atoi(opt["anchor_num"].c_str());
    info.classes = atoi(opt["classes"].c_str());
    info.max_boxes = atoi(opt["max_boxes"].c_str());
    info.net_width = atoi(opt["net_width"].c_str());
    info.net_height = atoi(opt["net_height"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.anchor_num, 0); // Mask
    if (opt.find("mask") == opt.end()) {
        for (int i = 0; i < info.anchor_num; ++i)
            kernel[0].weight[i] = i;
    } else {
        stringstream ss;
        ss << opt["mask"];
        int n;
        char c;
        for (int i = 0; i < info.anchor_num; ++i) {
            ss >> n >> c;
            kernel[0].weight[i] = n;
        }
    }
    info.kernel_num = 1;
    
    biases = new Tensor(1, 1, 2 * info.total_anchor_num, 0); // Anchor
    stringstream ss;
    ss << opt["anchor"];
    int w, h;
    char c;
    for (int i = 0; i < info.total_anchor_num; ++i) {
        ss >> w >> c >> h;
        biases->weight[i * 2 + 0] = w;
        biases->weight[i * 2 + 1] = h;
    }
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

Tensor* YOLOv3Layer::Forward(Tensor *input_tensor_, Forward_Args *args) {
    input_tensor = input_tensor_;
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    int channel_size = info.output_width * info.output_height;
    
    copy_cpu(info.output_number * info.batch_size, input, output);
    for (int b = 0; b < info.batch_size; ++b) {
        for (int n = 0; n < info.anchor_num; ++n) {
            int index = yolo_index(b, n, 0, 0, 0);
            activate_array(output + index, 2 * channel_size, SIGMOID);
            index = yolo_index(b, n, 0, 0, 4);
            activate_array(output + index, (1 + info.classes) * channel_size, SIGMOID);
        }
    }
    
    vector<Detection> dets = yolo_get_detection_without_correction();
    detection = Tensor(1, (info.classes + 5), (int)dets.size(), 0);
    float *value = detection.weight;
    for (int i = 0; i < (int)dets.size(); ++i) {
        *(value++) = dets[i].bbox.x;
        *(value++) = dets[i].bbox.y;
        *(value++) = dets[i].bbox.w;
        *(value++) = dets[i].bbox.h;
        *(value++) = dets[i].objectness;
        for (int c = 0; c < info.classes; ++c) {
            *(value++) = dets[i].prob[c];
        }
    }
    
    return &detection;
}

float YOLOv3Layer::Backward(vfloat& target) {
    float ignore_iou_threshold = 0.7; // TODO: should be input
    float truth_iou_threshold = 1.0;
    
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    float loss = 0;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    int target_size = 5 * info.classes * info.max_boxes;
    
    float *mask = kernel[0].weight;
    float *bias = biases->weight;
    
    float *output = output_tensor->weight;
    float *input_delta = input_tensor->delta_weight;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int j = 0; j < info.output_height; ++j) {
            for (int i = 0; i < info.output_width; ++i) {
                for (int n = 0; n < info.anchor_num; ++n) {
                    int box_index = yolo_index(b, n, i, j, 0);
                    Box pred = yolo_get_box(box_index, mask[n], i, j, width, height, info.net_width, info.net_height, channel_size);
                    float best_iou = 0;
                    int best_t = 0;
                    for(int t = 0; t < info.max_boxes; ++t){
                        Box truth = vfloat_to_box(target, 5 * t + b * target_size);
                        if(!truth.x)
                            break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = yolo_index(b, n, i, j, 4);
                    avg_anyobj += output[obj_index];
                    input_delta[obj_index] = 0 - output[obj_index];
                    if (best_iou > ignore_iou_threshold) {
                        input_delta[obj_index] = 0;
                    }
                    if (best_iou > truth_iou_threshold) {
                        input_delta[obj_index] = 1 - output[obj_index];

                        int cls = target[best_t * (4 + 1) + b * target_size + 4];
//                        if (l.map) class = l.map[class];
                        int class_index = yolo_index(b, n, i, j, 5);
                        delta_yolo_class(output, input_delta, class_index, cls, classes, channel_size, nullptr);
                        Box truth = vfloat_to_box(target, best_t * (4 + 1) + b * target_size);
                        delta_yolo_box(truth, output, bias, input_delta, mask[n], box_index, i, j, width, height, net_width, net_height, (2 - truth.w * truth.h), channel_size);
                    }
                }
            }
        }
        for(int t = 0; t < info.max_boxes; ++t){
            Box truth = vfloat_to_box(target, 5 * t + b * target_size);

            if(!truth.x)
                break;
            float best_iou = 0;
            int best_n = 0;
            int i = (truth.x * width);
            int j = (truth.y * height);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(int n = 0; n < info.total_anchor_num; ++n){
                Box pred;
                pred.w = bias[2 * n + 0] / net_width;
                pred.h = bias[2 * n + 1] / net_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            
            int mask_n = int_index(mask, best_n, info.anchor_num);
            if(mask_n >= 0){
                int box_index = yolo_index(b, mask_n, i, j, 0);
                float iou = delta_yolo_box(truth, output, bias, input_delta, best_n, box_index, i, j, width, height, net_width, net_height, (2 - truth.w * truth.h), channel_size);
                int obj_index = yolo_index(b, mask_n, i, j, 4);
                avg_obj += output[obj_index];
                input_delta[obj_index] = 1 - output[obj_index];

                int cls = target[t * (4 + 1) + b * target_size + 4];
//                if (l.map) class = l.map[class];
                int class_index = yolo_index(b, mask_n, i, j, 5);
                delta_yolo_class(output, input_delta, class_index, cls, classes, channel_size, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    loss = pow(mag_array(input_delta, info.output_number * info.batch_size), 2);
    
    return loss;
}

float YOLOv3Layer::mag_array(float *a, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i){
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

int YOLOv3Layer::int_index(float *a, int val, int n) {
    for(int i = 0; i < n; ++i) {
        if(a[i] == val)
            return i;
    }
    return -1;
}

void YOLOv3Layer::delta_yolo_class(float *feature, float *delta, int index, int cls, int classes, int stride, float *avg_cat) {
    if (delta[index]){
        delta[index + stride*cls] = 1 - feature[index + stride*cls];
        if(avg_cat)
            *avg_cat += feature[index + stride*cls];
        return;
    }
    for(int n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == cls) ? 1 : 0) - feature[index + stride * n];
        if(n == cls && avg_cat)
            *avg_cat += feature[index + stride*n];
    }
}

float YOLOv3Layer::delta_yolo_box(Box &truth, float *feature, float *bias, float *delta, int n, int index, int w, int h, int width, int height, int net_width, int net_height, float scale, int stride) {
    Box pred = yolo_get_box(index, n, w, h, width, height, net_width, net_height, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x * width - w);
    float ty = (truth.y * height - h);
    float tw = log(truth.w * w / bias[2 * n + 0]);
    float th = log(truth.h * h / bias[2 * n + 1]);

    delta[index + 0 * stride] = scale * (tx - feature[index + 0 * stride]);
    delta[index + 1 * stride] = scale * (ty - feature[index + 1 * stride]);
    delta[index + 2 * stride] = scale * (tw - feature[index + 2 * stride]);
    delta[index + 3 * stride] = scale * (th - feature[index + 3 * stride]);
    return iou;
}

int YOLOv3Layer::yolo_index(int b, int anchor_num, int w, int h, int entry) {
    int channel_size = info.output_width * info.output_height;
    int batch_offset = b * info.output_number;
    int anchor_offset = anchor_num * channel_size * (info.classes + 5);
    int entry_offset = entry * channel_size;
    int position_offset = h * info.output_width + w;
    return batch_offset + anchor_offset + entry_offset + position_offset;
}

vector<Detection> YOLOv3Layer::yolo_get_detection_without_correction() {
    float threshold = 0.5; // TODO: threshold input
    float *feature = output_tensor->weight;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    float *mask = kernel[0].weight;
    
    vector<Detection> dets;
    
    for (int n = 0; n < info.anchor_num; ++n) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int obj_index = yolo_index(0, n, w, h, 4);
                float objectness = feature[obj_index];
                if (objectness > threshold) {
                    Detection det; det.prob.reserve(classes);
                    int box_index = yolo_index(0, n, w, h, 0);
                    det.objectness = objectness;
                    det.bbox = yolo_get_box(box_index, mask[n], w, h, width, height, net_width, net_height, channel_size);
                    for (int c = 0; c < classes; ++c) {
                        int class_index = yolo_index(0, n, w, h, 5 + c);
                        float prob = objectness * feature[class_index];
                        det.prob.push_back((prob > threshold) ? prob : 0);
                    }
                    dets.push_back(det);
                }
            }
        }
    }
    return dets;
}

Box YOLOv3Layer::yolo_get_box(int index, int n, int w, int h, int width, int height, int net_width, int net_height, int stride) {
    Box b;
    float *x = output_tensor->weight;
    float *bias = biases->weight;
    
    b.x = (w + x[index + 0 * stride]) / width;
    b.y = (h + x[index + 1 * stride]) / height;
    b.w = exp(x[index + 2 * stride]) * bias[2 * n] / net_width;
    b.h = exp(x[index + 3 * stride]) * bias[2 * n + 1] / net_height;
    
    return b;
}
