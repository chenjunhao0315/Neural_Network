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
        case LayerType::Yolov3: delete yolov3_layer; break;
        case LayerType::Yolov4: delete yolov4_layer; break;
        case LayerType::Mish: delete mish_layer; break;
        case LayerType::Dropout: delete dropout_layer; break;
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
    yolov4_layer = nullptr;
    mish_layer = nullptr;
    dropout_layer = nullptr;
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
    yolov4_layer = nullptr;
    mish_layer = nullptr;
    dropout_layer = nullptr;
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
    yolov4_layer = nullptr;
    mish_layer = nullptr;
    dropout_layer = nullptr;
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
            case LayerType::Yolov3:
                yolov3_layer = new YOLOv3Layer(*L.yolov3_layer);
                break;
            case LayerType::Yolov4:
                yolov4_layer = new YOLOv4Layer(*L.yolov4_layer);
                break;
            case LayerType::Mish:
                mish_layer = new MishLayer(*L.mish_layer);
                break;
            case LayerType::Dropout:
                dropout_layer = new DropoutLayer(*L.dropout_layer);
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
    yolov4_layer = L.yolov4_layer;
    mish_layer = L.mish_layer;
    dropout_layer = L.dropout_layer;
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
    L.yolov4_layer = nullptr;
    L.mish_layer = nullptr;
    L.dropout_layer = nullptr;
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
    yolov4_layer = nullptr;
    mish_layer = nullptr;
    dropout_layer = nullptr;
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
            case LayerType::Yolov3:
                yolov3_layer = new YOLOv3Layer(*L.yolov3_layer);
                break;
            case LayerType::Yolov4:
                yolov4_layer = new YOLOv4Layer(*L.yolov4_layer);
                break;
            case LayerType::Mish:
                mish_layer = new MishLayer(*L.mish_layer);
                break;
            case LayerType::Dropout:
                dropout_layer = new DropoutLayer(*L.dropout_layer);
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
    yolov4_layer = nullptr;
    mish_layer = nullptr;
    dropout_layer = nullptr;
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
        case LayerType::Yolov3:
            yolov3_layer = new YOLOv3Layer(opt_); break;
        case LayerType::Yolov4:
            yolov4_layer = new YOLOv4Layer(opt_); break;
        case LayerType::Mish:
            mish_layer = new MishLayer(opt_); break;
        case LayerType::Dropout:
            dropout_layer = new DropoutLayer(opt_); break;
        default:
            fprintf(stderr, "[Model_Layer] Unknown layer type!\n"); break;
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
        return LayerType::Yolov3;
    } else if (type == "YOLOv4") {
        return LayerType::Yolov4;
    } else if (type == "Mish") {
        return LayerType::Mish;
    } else if (type == "Dropout") {
        return LayerType::Dropout;
    }
    return LayerType::Error;
}

void Model_Layer::Forward(Tensor* input_tensor_, bool train) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->Forward();
        case LayerType::Pooling:
            return pooling_layer->Forward();
        case LayerType::PRelu:
            return prelu_layer->Forward();
        case LayerType::Fullyconnected:
            return fullyconnected_layer->Forward();
        case LayerType::Relu:
            return relu_layer->Forward();
        case LayerType::ShortCut:
            return shortcut_layer->Forward();
        case LayerType::Softmax:
            return softmax_layer->Forward();
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Forward();
        case LayerType::Input:
            return input_layer->Forward(input_tensor_);
        case LayerType::LRelu:
            return lrelu_layer->Forward();
        case LayerType::Sigmoid:
            return sigmoid_layer->Forward();
        case LayerType::BatchNormalization:
            return batchnorm_layer->Forward(train);
        case LayerType::UpSample:
            return upsample_layer->Forward();
        case LayerType::Concat:
            return concat_layer->Forward();
        case LayerType::Yolov3:
            return yolov3_layer->Forward(train);
        case LayerType::Yolov4:
            return yolov4_layer->Forward(train);
        case LayerType::Mish:
            return mish_layer->Forward();
        case LayerType::Dropout:
            return dropout_layer->Forward();
        default:
            break;
    }
}

Tensor* Model_Layer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    switch (type) {
        case LayerType::Convolution:
            return convolution_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Pooling:
            return pooling_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::PRelu:
            return prelu_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Fullyconnected:
            return fullyconnected_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Relu:
            return relu_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::ShortCut:
            return shortcut_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Softmax:
            return softmax_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Input:
            return input_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::LRelu:
            return lrelu_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Sigmoid:
            return sigmoid_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::BatchNormalization:
            return batchnorm_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::UpSample:
            return upsample_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Concat:
            return concat_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Yolov3:
            return yolov3_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Yolov4:
            return yolov4_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Mish:
            return mish_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        case LayerType::Dropout:
            return dropout_layer->connectGraph(input_tensor_, extra_tensor_, workspace);
        default:
            break;
    }
    return nullptr;
}

float Model_Layer::Backward(Tensor *target) { 
    switch (type) {
        case LayerType::Softmax:
            return softmax_layer->Backward(target);
        case LayerType::EuclideanLoss:
            return euclideanloss_layer->Backward(target);
        case LayerType::Yolov3:
            return yolov3_layer->Backward(target);
        case LayerType::Yolov4:
            return yolov4_layer->Backward(target);
        case LayerType::Convolution:
            convolution_layer->Backward(); break;
        case LayerType::Pooling:
            pooling_layer->Backward(); break;
        case LayerType::PRelu:
            prelu_layer->Backward(); break;
        case LayerType::Input:
            input_layer->Backward(); break;
        case LayerType::Fullyconnected:
            fullyconnected_layer->Backward(); break;
        case LayerType::Relu:
            relu_layer->Backward(); break;
        case LayerType::ShortCut:
            shortcut_layer->Backward(); break;
        case LayerType::LRelu:
            lrelu_layer->Backward(); break;
        case LayerType::Sigmoid:
            sigmoid_layer->Backward(); break;
        case LayerType::BatchNormalization:
            batchnorm_layer->Backward(); break;
        case LayerType::UpSample:
            upsample_layer->Backward(); break;
        case LayerType::Concat:
            concat_layer->Backward(); break;
        case LayerType::Mish:
            mish_layer->Backward(); break;
        case LayerType::Dropout:
            dropout_layer->Backward(); break;
        default:
            break;
    }
    return 0;
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
    } else if (type == LayerType::Yolov3) {
        yolov3_layer->shape();
    } else if (type == LayerType::Yolov4) {
        yolov4_layer->shape();
    } else if (type == LayerType::Mish) {
        mish_layer->shape();
    } else if (type == LayerType::Dropout) {
        dropout_layer->shape();
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
        case LayerType::Yolov3:
            return yolov3_layer->getParameter(type_);
        case LayerType::Yolov4:
            return yolov4_layer->getParameter(type_);
        case LayerType::Mish:
            return mish_layer->getParameter(type_);
        case LayerType::Dropout:
            return dropout_layer->getParameter(type_);
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
    } else if (type == LayerType::Yolov3) {
        return yolov3_layer->save(f);
    } else if (type == LayerType::Yolov4) {
        return yolov4_layer->save(f);
    } else if (type == LayerType::Mish) {
        return mish_layer->save(f);
    } else if (type == LayerType::Dropout) {
        return dropout_layer->save(f);
    }
    return false;
}

bool Model_Layer::save_raw(FILE *f) {
    if (type == LayerType::Input) {
        return input_layer->save_raw(f);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->save_raw(f);
    } else if (type == LayerType::Relu) {
        return relu_layer->save_raw(f);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->save_raw(f);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->save_raw(f);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->save_raw(f);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->save_raw(f);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->save_raw(f);
    } else if (type == LayerType::ShortCut) {
        return shortcut_layer->save_raw(f);
    } else if (type == LayerType::LRelu) {
        return lrelu_layer->save_raw(f);
    } else if (type == LayerType::Sigmoid) {
        return sigmoid_layer->save_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        return batchnorm_layer->save_raw(f);
    } else if (type == LayerType::UpSample) {
        return upsample_layer->save_raw(f);
    } else if (type == LayerType::Concat) {
        return concat_layer->save_raw(f);
    } else if (type == LayerType::Yolov3) {
        return yolov3_layer->save_raw(f);
    } else if (type == LayerType::Yolov4) {
        return yolov4_layer->save_raw(f);
    } else if (type == LayerType::Mish) {
        return mish_layer->save_raw(f);
    } else if (type == LayerType::Dropout) {
        return dropout_layer->save_raw(f);
    }
    return false;
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
    } else if (type == LayerType::Yolov3) {
        return yolov3_layer->load(f);
    } else if (type == LayerType::Yolov4) {
        return yolov4_layer->load(f);
    } else if (type == LayerType::Mish) {
        return mish_layer->load(f);
    } else if (type == LayerType::Dropout) {
        return dropout_layer->load(f);
    }
    return false;
}

bool Model_Layer::load_raw(FILE *f) {
    if (type == LayerType::Input) {
        return input_layer->load_raw(f);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->load_raw(f);
    } else if (type == LayerType::Relu) {
        return relu_layer->load_raw(f);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->load_raw(f);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->load_raw(f);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->load_raw(f);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->load_raw(f);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->load_raw(f);
    } else if (type == LayerType::ShortCut) {
        return shortcut_layer->load_raw(f);
    } else if (type == LayerType::LRelu) {
        return lrelu_layer->load_raw(f);
    } else if (type == LayerType::Sigmoid) {
        return sigmoid_layer->load_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        return batchnorm_layer->load_raw(f);
    } else if (type == LayerType::UpSample) {
        return upsample_layer->load_raw(f);
    } else if (type == LayerType::Concat) {
        return concat_layer->load_raw(f);
    } else if (type == LayerType::Yolov3) {
        return yolov3_layer->load_raw(f);
    } else if (type == LayerType::Yolov4) {
        return yolov4_layer->load_raw(f);
    } else if (type == LayerType::Mish) {
        return mish_layer->load_raw(f);
    } else if (type == LayerType::Dropout) {
        return dropout_layer->load_raw(f);
    }
    return 0;
}

bool Model_Layer::to_prototxt(FILE *f, int refine_id, vector<LayerOption> &refine_struct, unordered_map<string, int> &id_table) {
    if (type == LayerType::Input) {
        return input_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Fullyconnected) {
        return fullyconnected_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Relu) {
        return relu_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Softmax) {
        return softmax_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Convolution) {
        return convolution_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Pooling) {
        return pooling_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::EuclideanLoss) {
        return euclideanloss_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::PRelu) {
        return prelu_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::ShortCut) {
        return shortcut_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::LRelu) {
        return lrelu_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Sigmoid) {
        return sigmoid_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::BatchNormalization) {
        return batchnorm_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::UpSample) {
        return upsample_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Concat) {
        return concat_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Yolov3) {
        return yolov3_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Yolov4) {
        return yolov4_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Mish) {
        return mish_layer->to_prototxt(f, refine_id, refine_struct, id_table);
    } else if (type == LayerType::Dropout) {
        return dropout_layer->to_prototxt(f, refine_id, refine_struct, id_table);
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
        case LayerType::Yolov3:
            return yolov3_layer->ClearGrad();
        case LayerType::Yolov4:
            return yolov4_layer->ClearGrad();
        case LayerType::Mish:
            return mish_layer->ClearGrad();
        case LayerType::Dropout:
            return dropout_layer->ClearGrad();
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
        case LayerType::Yolov3:
            return yolov3_layer->getTrainArgs();
        case LayerType::Yolov4:
            return yolov4_layer->getTrainArgs();
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
            return Train_Args(kernel, biases, info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension, 1, (info.batchnorm) ? 0 : info.output_dimension, vfloat{0, 1});
        case LayerType::Fullyconnected:
            return Train_Args(kernel, biases, info.input_number * info.output_dimension, 1, (info.batchnorm) ? 0 : info.output_dimension, vfloat{0, 1});
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
        opt = L->opt;
        input_tensor = L->input_tensor;
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
    }
}

BaseLayer::BaseLayer(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        opt = L.opt;
        input_tensor = L.input_tensor;
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
}

BaseLayer& BaseLayer::operator=(const BaseLayer &L) {
    if (this != &L) {
        type = L.type;
        name = L.name;
        input_name = L.input_name;
        info = L.info;
        opt = L.opt;
        input_tensor = L.input_tensor;
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
        case Yolov3: return "YOLOv3";
        case Yolov4: return "YOLOv4";
        case Mish: return "Mish";
        case Dropout: return "Dropout";
        case Error: return "Error";
    }
    return "Unknown";
}

Tensor* BaseLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return output_tensor;
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
    output_tensor->clearDeltaWeight();
}

bool BaseLayer::save(FILE *f) {
    int y = 1, n = 0;
    
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
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        fwrite((info.batchnorm) ? &y : &n, sizeof(int), 1, f);
        kernel[0].save(f);
        biases->save(f);
    } else if (type == LayerType::Softmax) {
        int input_number = info.input_number / info.output_dimension;
        fwrite(&input_number, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Convolution) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        fwrite(&info.kernel_width, sizeof(int), 1, f);
        fwrite(&info.kernel_height, sizeof(int), 1, f);
        fwrite(&info.stride, sizeof(int), 1, f);
        fwrite(&info.padding, sizeof(int), 1, f);
        fwrite((info.batchnorm) ? &y : &n, sizeof(int), 1, f);
        kernel[0].save(f);
        biases->save(f);
    } else if (type == LayerType::Relu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Pooling) {
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.kernel_width, sizeof(int), 1, f);
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
        fwrite(&info.shortcut_width, sizeof(int), 1, f);
        fwrite(&info.shortcut_height, sizeof(int), 1, f);
        fwrite(&info.shortcut_dimension, sizeof(int), 1, f);
        int len = (int)strlen(opt["shortcut"].c_str());
        fwrite(&len, sizeof(int), 1, f);
        char *output = new char [len];
        strcpy(output, opt["shortcut"].c_str());
        fwrite(output, sizeof(char), len, f);
    } else if (type == LayerType::LRelu) {
        fwrite(&info.output_width, sizeof(int), 1, f);
        fwrite(&info.output_height, sizeof(int), 1, f);
        fwrite(&info.output_dimension, sizeof(int), 1, f);
        kernel[0].save(f);
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
    } else if (type == LayerType::UpSample) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.stride, sizeof(int), 1, f);
        fwrite((info.reverse) ? &y : &n, sizeof(int), 1, f);
    } else if (type == LayerType::Concat) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.concat_num, sizeof(int), 1, f);
        for (int i = 1; i <= info.concat_num; ++i) {
            int concat_dim = kernel[0][i];
            fwrite(&info.input_width, sizeof(int), 1, f);
            fwrite(&info.input_height, sizeof(int), 1, f);
            fwrite(&concat_dim, sizeof(int), 1, f);
        }
        fwrite(&info.splits, sizeof(int), 1, f);
        fwrite(&info.split_id, sizeof(int), 1, f);
        int len = (int)strlen(opt["concat"].c_str());
        fwrite(&len, sizeof(int), 1, f);
        char output[len];
        strcpy(output, opt["concat"].c_str());
        fwrite(output, sizeof(char), len, f);
    } else if (type == LayerType::Yolov3) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.total_anchor_num, sizeof(int), 1, f);
        fwrite(&info.anchor_num, sizeof(int), 1, f);
        fwrite(&info.classes, sizeof(int), 1, f);
        fwrite(&info.max_boxes, sizeof(int), 1, f);
        fwrite(&info.net_width, sizeof(int), 1, f);
        fwrite(&info.net_height, sizeof(int), 1, f);
        kernel[0].save(f);
        biases->save(f);
    } else if (type == LayerType::Yolov4) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.total_anchor_num, sizeof(int), 1, f);
        fwrite(&info.anchor_num, sizeof(int), 1, f);
        fwrite(&info.classes, sizeof(int), 1, f);
        fwrite(&info.max_boxes, sizeof(int), 1, f);
        fwrite(&info.net_width, sizeof(int), 1, f);
        fwrite(&info.net_height, sizeof(int), 1, f);
        kernel[0].save(f);
        biases->save(f);
    } else if (type == LayerType::Mish) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
    } else if (type == LayerType::Dropout) {
        fwrite(&info.input_width, sizeof(int), 1, f);
        fwrite(&info.input_height, sizeof(int), 1, f);
        fwrite(&info.input_dimension, sizeof(int), 1, f);
        fwrite(&info.probability, sizeof(float), 1, f);
    }
    return true;
}

bool BaseLayer::load(FILE *f) {
    if (type == LayerType::Fullyconnected || type == LayerType::Convolution) {
        kernel[0].load(f);
        biases->load(f);
    } else if (type == LayerType::PRelu || type == LayerType::LRelu) {
        kernel->load(f);
    } else if (type == LayerType::BatchNormalization) {
        kernel[0].load(f);
        kernel[3].load(f);
        kernel[4].load(f);
        biases->load(f);
        // Test
        kernel[1] = kernel[3];
        kernel[2] = kernel[4];
    } else if (type == LayerType::Yolov3) {
        kernel[0].load(f);
        biases->load(f);
    } else if (type == LayerType::Yolov4) {
        kernel[0].load(f);
        biases->load(f);
    }
    return true;
}

bool BaseLayer::save_raw(FILE *f) {
    if (type == LayerType::Fullyconnected || type == LayerType::Convolution) {
        if (!info.batchnorm) {
            biases->save_raw(f);
        }
        kernel[0].save_raw(f);
    } else if (type == LayerType::PRelu) {
        kernel->save_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        biases->save_raw(f);
        kernel[0].save_raw(f);
        kernel[3].save_raw(f);
        kernel[4].save_raw(f);
    }
    return true;
}

bool BaseLayer::load_raw(FILE *f) {
    if (type == LayerType::Fullyconnected || type == LayerType::Convolution) {
        if (!info.batchnorm) {
            biases->load_raw(f);
        }
        kernel[0].load_raw(f);
    } else if (type == LayerType::PRelu) {
        kernel->load_raw(f);
    } else if (type == LayerType::BatchNormalization) {
        biases->load_raw(f);
        kernel[0].load_raw(f);
        kernel[3].load_raw(f);
        kernel[4].load_raw(f);
        // Test
        kernel[1] = kernel[3];
        kernel[2] = kernel[4];
    }
    return true;
}

bool BaseLayer::to_prototxt(FILE *f, int refine_id, vector<LayerOption> &refine_struct, unordered_map<string, int> &id_table) {
    if (type == LayerType::Input) {
        fprintf(f, "input: \"data\"\n");
        fprintf(f, "input_dim: %d\n", info.batch_size);
        fprintf(f, "input_dim: %d\n", info.output_dimension);
        fprintf(f, "input_dim: %d\n", info.output_width);
        fprintf(f, "input_dim: %d\n", info.output_height);
    } else {
        fprintf(f, "layer {\n");
        fprintf(f, "  name: \"%s\"\n", name.c_str());
        fprintf(f, "  type: \"%s\"\n", type_to_string().c_str());
        fprintf(f, "  bottom: \"%s\"\n", refine_struct[refine_id]["input_name"].c_str());
        fprintf(f, "  top: \"%s\"\n", refine_struct[refine_id]["name"].c_str());
        
        if (type == LayerType::Fullyconnected) {
            fprintf(f, "  connected_param {\n");
            fprintf(f, "    num_output: %d\n", info.output_dimension);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Softmax) {
        } else if (type == LayerType::Convolution) {
            fprintf(f, "  convolution_param {\n");
            fprintf(f, "    num_output: %d\n", info.output_dimension);
            fprintf(f, "    kernel_size: %d\n", info.kernel_width);
            fprintf(f, "    pad: %d\n", info.padding);
            fprintf(f, "    stride: %d\n", info.stride);
            if (info.batchnorm) {
                fprintf(f, "    bias_term: false\n");
            } else {
                fprintf(f, "    bias_term: true\n");
            }
            fprintf(f, "  }\n");
        } else if (type == LayerType::Relu) {
        } else if (type == LayerType::Pooling) {
            fprintf(f, "  pool_param {\n");
            fprintf(f, "    pool: MAX\n");
            fprintf(f, "    kernel_size: %d\n", info.kernel_width);
            fprintf(f, "    stride: %d\n", info.stride);
            fprintf(f, "  }\n");
        } else if (type == LayerType::EuclideanLoss) {
        } else if (type == LayerType::PRelu) {
        } else if (type == LayerType::ShortCut) {
            fprintf(f, "  bottom: \"%s\"\n", refine_struct[id_table[opt["shortcut"]]]["name"].c_str());
        } else if (type == LayerType::LRelu) {
            fprintf(f, "  lrelu_param {\n");
            fprintf(f, "    negative_slope: %g\n", kernel[0][0]);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Sigmoid) {
        } else if (type == LayerType::BatchNormalization) {
        } else if (type == LayerType::UpSample) {
        } else if (type == LayerType::Concat) {
            for (int i = 1; i <= info.concat_num; ++i) {
                fprintf(f, "  bottom: \"%s\"\n", refine_struct[id_table[opt["concat_" + to_string(i) + "_name"]]]["name"].c_str());
            }
            fprintf(f, "  concat_param {\n");
            fprintf(f, "    splits: %d\n", info.splits);
            fprintf(f, "    split_id: %d\n", info.split_id);
            fprintf(f, "  }\n");
        } else if (type == LayerType::Yolov3) {
            fprintf(f, "  yolov3_param {\n");
            fprintf(f, "    classes: %d\n", info.classes);
            fprintf(f, "    total_anchor_num: %d\n", info.total_anchor_num);
            fprintf(f, "    anchor_num: %d\n", info.anchor_num);
            fprintf(f, "    anchor: \"%s\"\n", opt["anchor"].c_str());
            fprintf(f, "    mask: \"%s\"\n", opt["mask"].c_str());
            fprintf(f, "  }\n");
        } else if (type == LayerType::Yolov4) {
            fprintf(f, "  yolov4_param {\n");
            fprintf(f, "    classes: %d\n", info.classes);
            fprintf(f, "    total_anchor_num: %d\n", info.total_anchor_num);
            fprintf(f, "    anchor_num: %d\n", info.anchor_num);
            fprintf(f, "    anchor: \"%s\"\n", opt["anchor"].c_str());
            fprintf(f, "    mask: \"%s\"\n", opt["mask"].c_str());
            fprintf(f, "  }\n");
        } else if (type == LayerType::Mish) {
        } else if (type == LayerType::Dropout) {
            fprintf(f, "  dropout_param {\n");
            fprintf(f, "    ratioi: %g\n", info.probability);
            fprintf(f, "  }\n");
        }
        fprintf(f, "}\n");
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
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

void InputLayer::Forward(Tensor *input_tensor_) {
    input_tensor = input_tensor_;
    copy_cpu(info.output_number * info.batch_size, input_tensor->weight, output_tensor->weight);
}

ConvolutionLayer::ConvolutionLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Convolution;
    name = (opt.find("name") == opt.end()) ? "conv" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.kernel_width = atoi(opt["kernel_width"].c_str());
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.kernel_height = (opt.find("kernel_height") == opt.end()) ? info.kernel_width : atoi(opt["kernel_height"].c_str());
    info.stride = (opt.find("stride") == opt.end()) ? 1 : atoi(opt["stride"].c_str());
    info.padding = (opt.find("padding") == opt.end()) ? 0 : ((opt["padding"] == "same") ? ((info.kernel_width - 1) / 2) : atoi(opt["padding"].c_str()));
    
    info.output_width = (info.input_width + info.padding * 2 - info.kernel_width) / info.stride + 1;
    info.output_height = (info.input_height + info.padding * 2 - info.kernel_height) / info.stride + 1;
    info.output_dimension = atoi(opt["number_kernel"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.batchnorm = (opt.find("batchnorm") != opt.end());
    
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension, 0);
    float *kernel_weight = kernel->weight;
    float scale = sqrt(2.0 / (info.kernel_width * info.kernel_height * info.input_dimension));
    for (int i = 0; i < info.kernel_width * info.kernel_height * info.input_dimension * info.output_dimension; ++i) {
        *(kernel_weight++) = randn(0.0, scale);
    }
    info.kernel_num = 1;
    
    biases = new Tensor(1, 1, info.output_dimension, bias);
    
    int kernel_n = info.kernel_width * info.kernel_height * info.input_dimension;
    int out_size = info.output_width * info.output_height;
    int workspace_size = out_size * kernel_n;
    info.workspace_size = workspace_size;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* ConvolutionLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace_) {
    input_tensor = input_tensor_;
    workspace = workspace_;
    return output_tensor;
}

void ConvolutionLayer::Forward() {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *weights = kernel->weight;
    float *bias = biases->weight;
    
    int batch_size = info.batch_size;
    int input_number = info.input_number;
    int output_number = info.output_number;
    int output_size = info.output_width * info.output_height;
    int kernel_number = info.kernel_width * info.kernel_height * info.input_dimension;
    int kernel_size = info.kernel_width;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    
    int stride = info.stride;
    int padding = info.padding;
    
    fill_cpu(output_number * batch_size, output, 0);
    
    int m = info.output_dimension;
    int k = kernel_number;
    int n = output_size;
    for(int i = 0; i < batch_size; ++i){
        float *a = weights;
        float *b = workspace;
        float *c = output + i * n * m;
        float *im = input + i * input_number;

        if (kernel_size == 1) {
            // Doesn't need to rearrange
            b = im;
        } else {
            // Rearrange input tensor and put it into workspace
            im2col_cpu(im, input_dimension, input_height, input_width, kernel_size, stride, padding, b);
        }
        // Matrix multiplication (input with kernel) (Both not transpose)
        // Calculate output (input_channel * kernel_n) dot (kernel_n * output_channel)
        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    }

    if(!info.batchnorm){
        // If not batchnorm, add bias
        add_bias(output, bias, batch_size, info.output_dimension, output_size);
    }
}

void ConvolutionLayer::Backward() {
    float *input = input_tensor->weight;
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *bias_delta = biases->delta_weight;
    float *weights = kernel->weight;
    float *weights_delta = kernel->delta_weight;
    
    int batch_size = info.batch_size;
    int kernel_size = info.kernel_width;
    int stride = info.stride;
    int padding = info.padding;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    int input_number = info.input_number;
    
    
    int m = info.output_dimension;
    int n = info.kernel_width * info.kernel_height * info.input_dimension;
    int k = info.output_width * info.output_height;

    if(!info.batchnorm){
        // If not batchnorm, backpropagate bias
        backward_bias(bias_delta, output_delta, batch_size, info.output_dimension, k);
    }

    for(int i = 0; i < batch_size; ++i){
        float *a = output_delta + i * m * k;
        float *b = workspace;
        float *c = weights_delta;

        float *im  = input + i * input_number;
        float *imd = input_delta + i * input_number;

        if(kernel_size == 1){
            // Doesn't need to rearrange
            b = im;
        } else {
            // Rearrange input tensor and put it into workspace
            im2col_cpu(im, input_dimension, input_height, input_width, kernel_size, stride, padding, b);
        }
        // Matrix multiplication (input with output_delta) (input transpose)
        // Calculate weight_delta
        gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
        a = weights;
        b = output_delta + i * m * k;
        c = workspace;
        if (kernel_size == 1) {
            c = imd;
        }
        // Matrix multiplication (weight with output_delta) (weight transpose)
        // Calculate input_delta
        gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
        if (kernel_size != 1) {
            // Restore arrangement
            col2im_cpu(workspace, input_dimension, input_height, input_width, kernel_size, stride, padding, imd);
        }
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
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, info.output_number * info.batch_size, 0);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

void PoolingLayer::Forward() {
    int stride = info.stride;

    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    int input_width = info.input_width;
    int input_height = info.input_height;
    int input_dimension = info.input_dimension;
    int kernel_size = info.kernel_width;

    int w_offset = -info.padding / 2;
    int h_offset = -info.padding / 2;

    int h = info.output_height;
    int w = info.output_width;
    int c = info.output_dimension;
    
    float *indexes = kernel->weight;

    for(int b = 0; b < info.batch_size; ++b){
        for(int k = 0; k < c; ++k){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -10000000;
                    int max_i = -1;
                    for(int n = 0; n < kernel_size; ++n){
                        for(int m = 0; m < kernel_size; ++m){
                            int cur_h = h_offset + i*stride + n;
                            int cur_w = w_offset + j*stride + m;
                            int index = cur_w + input_width * (cur_h + input_height * (k + b * input_dimension));
                            bool valid = (cur_h >= 0 && cur_h < input_height && cur_w >= 0 && cur_w < input_width);
                            float val = (valid != false) ? input[index] : -10000000;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    output[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void PoolingLayer::Backward() {
    float *output_delta = output_tensor->delta_weight;
    float *input_delta = input_tensor->delta_weight;
    float *indexes = kernel->weight;
    
    int total_size = info.output_width * info.output_height * info.output_dimension * info.batch_size;
    for(int i = 0; i < total_size; ++i){
        int index = indexes[i];
        input_delta[index] += output_delta[i];
    }
}

FullyConnectedLayer::FullyConnectedLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Fullyconnected;
    name = (opt.find("name") == opt.end()) ? "fc" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.input_width = atoi(opt["input_width"].c_str());
    info.input_height = atoi(opt["input_height"].c_str());
    info.input_dimension = atoi(opt["input_dimension"].c_str());
    info.input_number = info.input_width * info.input_height * info.input_dimension;
    info.output_width = 1;
    info.output_height = 1;
    info.output_dimension = atoi(opt["number_neurons"].c_str());
    info.output_number = info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    info.batchnorm = (opt.find("batchnorm") != opt.end());
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.input_number, 1, info.output_dimension, 0);
    float *kernel_weight = kernel->weight;
    float scale = sqrt(1.0 / (info.input_number));
    for (int i = 0; i < info.input_number * info.output_dimension; ++i) {
        *(kernel_weight++) = randn(0.0, scale);
    }
    info.kernel_num = 1;
    
    float bias = (opt.find("bias") == opt.end()) ? 0 : atof(opt["bias"].c_str());
    biases = new Tensor(1, 1, info.output_dimension, bias);
    output_tensor = new Tensor(1, 1, info.output_dimension * info.batch_size, 0);
}

void FullyConnectedLayer::Forward() {
    fill_cpu(info.output_dimension * info.batch_size, output_tensor->weight, 0);
    
    int m = info.batch_size;
    int k = info.input_number;
    int n = info.output_number;
    float *a = input_tensor->weight;
    float *b = kernel->weight;
    float *c = output_tensor->weight;
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
    if(!info.batchnorm){
        add_bias(output_tensor->weight, biases->weight, info.batch_size, info.output_dimension, 1);
    }
}

void FullyConnectedLayer::Backward() {
    if(!info.batchnorm){
        backward_bias(biases->delta_weight, output_tensor->delta_weight, info.batch_size, info.output_dimension, 1);
    }

    int m = info.output_dimension;
    int k = info.batch_size;
    int n = info.input_number;
    float *a = output_tensor->delta_weight;
    float *b = input_tensor->weight;
    float *c = kernel->delta_weight;
    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = info.batch_size;
    k = info.output_dimension;
    n = info.input_number;

    a = output_tensor->delta_weight;
    b = kernel->weight;
    c = input_tensor->delta_weight;

    gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
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

void ReluLayer::Forward() {
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    float value;
    
    for (int i = info.input_number * info.batch_size; i--; ) {
        value = *(val++);
        *(out++) = (value < 0) ? 0 : value;
    }
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

void PReluLayer::Forward() {
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

void SoftmaxLayer::Forward() {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *act = input_tensor->weight;
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
}

float SoftmaxLayer::Backward(Tensor *target) {
    int one_batch_input_size = info.input_number;
    int one_batch_output_size = info.output_dimension;
    
    float *cal_delta_weight = input_tensor->delta_weight;
    float *expo_sum_ptr = kernel->weight;
    float *target_ptr = target->weight;
    
    int output_dimension = info.output_dimension;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < output_dimension; ++i) {
            float indicator = (i == target_ptr[b]) ? 1.0 : 0.0;
            float mul = -(indicator - expo_sum_ptr[i]);
            cal_delta_weight[i] = mul;
        }
        cal_delta_weight += one_batch_input_size;
        expo_sum_ptr += one_batch_output_size;
    }
    float loss = 0;
    expo_sum_ptr = kernel->weight;
    for (int b = 0; b < info.batch_size; ++b) {
        loss += -log(expo_sum_ptr[(int)target_ptr[b]]);
        expo_sum_ptr += one_batch_output_size;
    }
    return loss;
}

EuclideanLossLayer::EuclideanLossLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::EuclideanLoss;
    name = (opt.find("name") == opt.end()) ? "el" : opt["name"];
    input_name = (opt.find("input_name") == opt.end()) ? "default" : opt["input_name"];
    
    info.output_width = atoi(opt["input_width"].c_str());
    info.output_height = atoi(opt["input_height"].c_str());
    info.output_dimension = atoi(opt["input_dimension"].c_str());
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    info.batch_size = atoi(opt["batch_size"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

void EuclideanLossLayer::Forward() {
    copy_cpu(info.output_number * info.batch_size, input_tensor->weight, output_tensor->weight);
}

float EuclideanLossLayer::Backward(Tensor *target) {
    Tensor *cal_tensor = input_tensor;
    
    int one_batch_size = info.output_dimension;
    
    float *cal_weight = cal_tensor->weight;
    float *cal_delta_weight = cal_tensor->delta_weight;
    float *target_ptr = target->weight;
    float loss = 0;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i < info.output_dimension; ++i) {
            float delta = cal_weight[i] - target_ptr[i + b * one_batch_size];
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
    info.shortcut_width = atoi(opt["shortcut_width"].c_str());
    info.shortcut_height = atoi(opt["shortcut_height"].c_str());
    info.shortcut_dimension = atoi(opt["shortcut_dimension"].c_str());
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

Tensor* ShortCutLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    shortcut_tensor = extra_tensor_[0];
    return output_tensor;
}

void ShortCutLayer::Forward() {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *shortcut = shortcut_tensor->weight;
    
    int shortcut_width = info.shortcut_width;
    int shortcut_height = info.shortcut_height;
    int shortcut_dimension = info.shortcut_dimension;
    
    copy_cpu(info.input_number * info.batch_size, input, output);
    shortcut_cpu(info.batch_size, shortcut_width, shortcut_height, shortcut_dimension, shortcut, info.output_width, info.output_height, info.output_dimension, 1, 1, output);
}

void ShortCutLayer::Backward() {
    
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(input_delta++) += *(output_delta++);
    }
    
    int shortcut_width = info.shortcut_width;
    int shortcut_height = info.shortcut_height;
    int shortcut_dimension = info.shortcut_dimension;
    
    float *shortcut_delta = shortcut_tensor->delta_weight;
    output_delta = output_tensor->delta_weight;
    
    shortcut_cpu(info.batch_size, info.output_width, info.output_height, info.output_dimension, output_delta, shortcut_width, shortcut_height, shortcut_dimension, 1, 1, shortcut_delta);
}

void ShortCutLayer::shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    int stride = w1 / w2;
    int sample = w2 / w1;
//    assert(stride == h1 / h2);
//    assert(sample == h2 / h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < minc; ++k){
            for(int j = 0; j < minh; ++j){
                for(int i = 0; i < minw; ++i){
                    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
                }
            }
        }
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
    
    float alpha = (opt.find("alpha") == opt.end()) ? 0.1 : atof(opt["alpha"].c_str());
    kernel = new Tensor [1];
    kernel[0] = Tensor(1, 1, 1, alpha);
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0.0);
}

void LReluLayer::Forward() {
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    float value, scale = kernel[0][0];
    for (int i = info.input_number * info.batch_size; i--; ++val) {
        value = *val;
        *(out++) = (value > 0) ? value : value * scale;
    }
}

void LReluLayer::Backward() {
    int one_batch_size = info.input_number;
    float *act_weight = output_tensor->weight;
    float *act_grad = output_tensor->delta_weight;
    float *pos_grad = input_tensor->delta_weight;
    float chain_grad, scale = kernel[0][0];
    
    for (int b = info.batch_size; b--; ) {
        for (int i = 0; i < info.input_number; ++i) {
            chain_grad = act_grad[i];
            if (act_weight[i] < 0) {
                pos_grad[i] += chain_grad * scale;
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

void SigmoidLayer::Forward() {
    float *val = input_tensor->weight;
    float *out = output_tensor->weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        *(out++) = 1.0 / (1.0 + exp(-(*(val++))));
    }
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

void BatchNormalizationlayer::Forward(bool train) {
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
    
    if (train) {
        mean_cpu(output, batch_size, output_dimension, channel_size, mean);
        variance_cpu(output, mean, batch_size, output_dimension, channel_size, variance);

        scal_cpu(output_dimension, 0.99, running_mean);
        axpy_cpu(output_dimension, 0.01, mean, running_mean);
        scal_cpu(output_dimension, 0.99, running_variance);
        axpy_cpu(output_dimension, 0.01, variance, running_variance);

        normalize(output, mean, variance, batch_size, output_dimension, channel_size);
        copy_cpu(total_size, output, x_norm);
    } else {
        normalize(output, running_mean, running_variance, batch_size, output_dimension, channel_size);
    }
    
    scale_bias(output, scale, batch_size, output_dimension, channel_size);
    add_bias(output, bias, batch_size, output_dimension, channel_size);
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

//    backward_bias(bias_delta, output_delta, batch_size, output_dimension, channel_size);
//    backward_scale_cpu(x_norm, output_delta, batch_size, output_dimension, channel_size, scale_delta);
//
//    scale_bias(output_delta, scale, batch_size, output_dimension, channel_size);
//
//    mean_delta_cpu(output_delta, variance, batch_size, output_dimension, channel_size, mean_delta);
//    variance_delta_cpu(x, output_delta, mean, variance, batch_size, output_dimension, channel_size, variance_delta);
//    normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch_size, output_dimension, channel_size, output_delta);
//    copy_cpu(one_batch_size * batch_size, output_delta, input_delta);
}

void BatchNormalizationlayer::backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates) {
    for(int f = 0; f < n; ++f){
        float sum = 0;
        for(int b = 0; b < batch; ++b){
            for(int i = 0; i < size; ++i){
                int index = i + size * (f + n * b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void BatchNormalizationlayer::mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta) {
    for(int i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (int j = 0; j < batch; ++j) {
            for (int k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void BatchNormalizationlayer::variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta) {
    for(int i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(int j = 0; j < batch; ++j){
            for(int k = 0; k < spatial; ++k){
                int index = j * filters * spatial + i * spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -0.5 * pow(variance[i] + 0.00001, (float)(-1.5));
    }
}
void BatchNormalizationlayer::normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta) {
    for(int j = 0; j < batch; ++j){
        for(int f = 0; f < filters; ++f){
            for(int k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
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

void UpSampleLayer::Forward() {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    
    if (info.reverse) {
        
    } else {
        upsample(input, info.input_width, info.input_height, info.input_dimension, info.batch_size, info.stride, true, 1, output);
    }
}

void UpSampleLayer::Backward() {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    
    if (info.reverse) {
        
    } else {
        upsample(input_delta, info.input_width, info.input_height, info.input_dimension, info.batch_size, info.stride, false, 1, output_delta);
    }
}

void UpSampleLayer::upsample(float *in, int w, int h, int c, int batch, int stride, bool forward, float scale, float *out) {
    int input_size = info.input_number;
    int input_channel_size = info.input_width * info.input_height;
    int output_size = info.output_number;
    int output_channel_size = info.output_width * info.output_height;
    
    for(int b = 0; b < batch; ++b){
        for(int k = 0; k < c; ++k){
            for(int j = 0; j < h * stride; ++j){
                for(int i = 0; i < w * stride; ++i){
                    int in_index = b * input_size + k * input_channel_size + (j / stride) * w + i / stride;
                    int out_index = b * output_size + k * output_channel_size + j * w * stride + i;
                    if(forward)
                        out[out_index] = scale * in[in_index];
                    else
                        in[in_index] += scale * out[out_index];
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
    info.output_dimension = info.input_dimension;
    
    info.concat_num = atoi(opt["concat_num"].c_str());
    kernel = new Tensor [1];    // Concat tensor structure
    kernel[0] = Tensor(1, 1, info.concat_num + 1, 0);
    kernel[0] = {float(info.input_dimension)};
    info.kernel_num = 1;
    
    for (int i = 1; i <= info.concat_num; ++i) {
        int width_check = atoi(opt[("concat_" + to_string(i) + "_width")].c_str());
        int height_check = atoi(opt[("concat_" + to_string(i) + "_height")].c_str());
        assert(width_check == info.input_width);
        assert(height_check == info.input_height);
        int concat_dimension = kernel[0][i] = atoi(opt[("concat_" + to_string(i) + "_dimension")].c_str());
        info.output_dimension += concat_dimension;
    }
    
    info.splits = (opt.find("splits") == opt.end()) ? 1 : atoi(opt["splits"].c_str());
    info.split_id = (opt.find("split_id") == opt.end()) ? 0 : atoi(opt["split_id"].c_str());
    
    info.output_dimension /= info.splits;
    info.output_number = info.output_width * info.output_height * info.output_dimension;
    
//    concat_tensor = new Tensor* [info.concat_num + 1];  // Concat tensor store
    concat_tensor.resize(info.concat_num + 1);
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

Tensor* ConcatLayer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    for (int i = 0; i <= info.concat_num; ++i) {
        concat_tensor[i] = (extra_tensor_[i]);
    }
    return output_tensor;
}

void ConcatLayer::Forward() {
    float *output = output_tensor->weight;
    
    int channel_size = info.input_width * info.input_height;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i <= info.concat_num; ++i) {
            int concat_input_size = channel_size * int(kernel[0][i]);
            int split_concat_input_size = concat_input_size / info.splits;
            float *concat = concat_tensor[i]->weight + b * concat_input_size + split_concat_input_size * info.split_id;
            for (int j = split_concat_input_size; j--; ) {
                *(output++) = *(concat++);
            }
        }
    }
}

void ConcatLayer::Backward() {
    float *output_delta = output_tensor->delta_weight;
    int check = output_tensor->size;
    
    int channel_size = info.input_width * info.input_height;
    
    for (int b = 0; b < info.batch_size; ++b) {
        for (int i = 0; i <= info.concat_num; ++i) {
            int concat_input_size = channel_size * int(kernel[0][i]);
            int split_concat_input_size = concat_input_size / info.splits;
            float *concat_delta = concat_tensor[i]->delta_weight + b * concat_input_size + split_concat_input_size * info.split_id;
            for (int j = split_concat_input_size; j--; ) {
                *(concat_delta++) += *(output_delta++);
                check--;
            }
        }
    }
}

MishLayer::MishLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Mish;
    name = (opt.find("name") == opt.end()) ? "mish" : opt["name"];
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
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);    // Input Storage
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

void MishLayer::Forward() {
    const float MISH_THRESHOLD = 20;
    float *src = input_tensor->weight;
    float *activation_input = kernel[0].weight;
    float *dst = output_tensor->weight;
    for (int i = info.input_number * info.batch_size; i--; ) {
        float x_val = *(src++);
        *(activation_input++) = x_val;
        *(dst++) = x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD));
    }
}

void MishLayer::Backward() {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *activation_input = kernel[0].weight;
    const float MISH_THRESHOLD = 20.0f;
    
    for (int i = info.output_number * info.batch_size; i--; ) {

        // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
        // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
        float inp = *(activation_input++);
        float sp = softplus_activate(inp, MISH_THRESHOLD);
        float grad_sp = 1 - exp(-sp);
        float tsp = tanh(sp);
        float grad_tsp = (1 - tsp * tsp) * grad_sp;
        float grad = inp * grad_tsp + tsp;
        *(input_delta++) += grad * *(output_delta++);
    }
}

DropoutLayer::DropoutLayer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Dropout;
    name = (opt.find("name") == opt.end()) ? "dropout" : opt["name"];
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
    
    info.probability = (opt.find("probability") == opt.end()) ? 0.5 : atof(opt["probability"].c_str());
    info.scale = 1.0 / (1.0 - info.probability);
    
    kernel = new Tensor [1];
    kernel[0] = Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);    // Probability storage
    info.kernel_num = 1;
    
    output_tensor = new Tensor(info.output_width, info.output_height, info.output_dimension * info.batch_size, 0);
}

void DropoutLayer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    float *prob = kernel->weight;
    float probability = info.probability;
    float scale = info.scale;
    
    if (!train) {
        copy_cpu(info.input_number * info.batch_size, input, output);
        return;
    }
    
    for(int i = info.input_number * info.batch_size; i--; ++input, ++output){
        float p = Random(0, 1);
        *(prob++) = p;
        if(p < probability)
            *(output) = 0;
        else
            *(output) = *(input) * scale;
    }
}

void DropoutLayer::Backward() {
    float *input_delta = input_tensor->delta_weight;
    float *output_delta = output_tensor->delta_weight;
    float *prob = kernel->weight;
    float probability = info.probability;
    float scale = info.scale;

    for(int i = info.input_number * info.batch_size; i--; ) {
        float p = *(prob++);
        if(p < probability)
            *(input_delta) = 0;
        else
            *(input_delta) = *(output_delta) * scale;
    }
}

YOLOv3Layer::YOLOv3Layer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Yolov3;
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
    info.ignore_iou_threshold = (opt.find("ignore_iou_threshold") == opt.end()) ? 0.5 : atof(opt["ignore_iou_threshold"].c_str());
    info.truth_iou_threshold = (opt.find("truth_iou_threshold") == opt.end()) ? 1 : atof(opt["truth_iou_threshold"].c_str());
    
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
    
    biases = new Tensor(2, 1, info.total_anchor_num, 0); // Anchor
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
    detection = Tensor(1, 1, 1, 0);
}

Tensor* YOLOv3Layer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return &detection;
}

void YOLOv3Layer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    int channel_size = info.output_width * info.output_height;
    
    memcpy(output, input, info.output_number * info.batch_size * sizeof(float));

    for (int b = 0; b < info.batch_size; ++b){
        for(int n = 0; n < info.anchor_num; ++n) {
            int index = entry_index(b, n * channel_size, 0);
            activate_array(output + index, 2 * channel_size, LOGISTIC);
            index = entry_index(b, n * channel_size, 4);
            activate_array(output + index, (1 + info.classes) * channel_size, LOGISTIC);
        }
    }
    input_tensor->clearDeltaWeight();
    
    if (!train) {
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
    }
}

float YOLOv3Layer::Backward(Tensor *target) {
    float ignore_iou_threshold = 0.7; // TODO: should be input
    float truth_iou_threshold = 1;
    
    int width = info.input_width;
    int height = info.input_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int anchor_num = info.anchor_num;
    int total_anchor_num = info.total_anchor_num;
    int channel_size = width * height;
    int max_boxes = info.max_boxes;
    int target_size = 5 * max_boxes;
    int classes = info.classes;
    
    float *output = output_tensor->weight;
    float *delta = input_tensor->delta_weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    float *target_ptr = target->weight;
    
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    int i,j,b,t,n;
    
    float loss = 0;
    for (b = 0; b < info.batch_size; ++b) {
        for (j = 0; j < height; ++j) {
            for (i = 0; i < width; ++i) {
                for (n = 0; n < anchor_num; ++n) {
                    int box_index = entry_index(b, n * channel_size + j * width + i, 0);
                    Box pred = get_yolo_box(output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, channel_size);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < max_boxes; ++t){
                        Box truth = float_to_box(target_ptr + t * (4 + 1) + target_size, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(b, n * channel_size + j * width + i, 4);
                    avg_anyobj += output[obj_index];
                    delta[obj_index] = output[obj_index];
                    if (best_iou > ignore_iou_threshold) {
                        delta[obj_index] = 0;
                    }
                    if (best_iou > truth_iou_threshold) {
                        delta[obj_index] = output[obj_index] - 1;

                        int cls = target_ptr[best_t*(4 + 1) + b * target_size + 4];
                        int class_index = entry_index(b, n * channel_size + j * width + i, 4 + 1);
                        delta_yolo_class(output, delta, class_index, cls, classes, channel_size, 0);
                        Box truth = float_to_box(target_ptr + best_t * (4 + 1) + b * target_size, 1);
                        delta_yolo_box(truth, output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);
                    }
                }
            }
        }
        for(t = 0; t < max_boxes; ++t){
            Box truth = float_to_box(target_ptr + t * (4 + 1) + b * target_size, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * width);
            j = (truth.y * height);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < total_anchor_num; ++n){
                Box pred = {0};
                pred.w = bias[2 * n + 0] / net_width;
                pred.h = bias[2 * n + 1] / net_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(mask, best_n, anchor_num);
            if(mask_n >= 0){
                int box_index = entry_index(b, mask_n * channel_size + j * width + i, 0);
                float iou = delta_yolo_box(truth, output, bias, best_n, box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);

                int obj_index = entry_index(b, mask_n * channel_size + j * width + i, 4);
                avg_obj += output[obj_index];
                delta[obj_index] = output[obj_index] - 1;

                int cls = target_ptr[t * (4 + 1) + b * target_size + 4];
                int class_index = entry_index(b, mask_n * channel_size + j * width + i, 4 + 1);
                delta_yolo_class(output, delta, class_index, cls, classes, channel_size, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    loss = pow(mag_array(delta, info.output_number * info.batch_size), 2);
    printf("Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(channel_size*anchor_num*info.batch_size), recall/count, recall75/count, count);
    return loss;
}

void YOLOv3Layer::delta_yolo_class(float *output, float *delta, int index, int cls, int classes, int stride, float *avg_cat) {
    if (delta[index]) {
        delta[index + stride * cls] = output[index + stride * cls] - 1;
        if(avg_cat)
            *avg_cat += output[index + stride * cls];
        return;
    }
    for(int n = 0; n < classes; ++n) {
        delta[index + stride * n] = output[index + stride * n] - ((n == cls) ? 1 : 0);
        if(n == cls && avg_cat)
            *avg_cat += output[index + stride * n];
    }
}

float YOLOv3Layer::delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride) {
    Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);
    
    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n + 0]);
    float th = log(truth.h * h / biases[2 * n + 1]);
    
    delta[index + 0 * stride] = -scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = -scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = -scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = -scale * (th - x[index + 3 * stride]);
    return iou;
}

int YOLOv3Layer::entry_index(int batch, int location, int entry) {
    int channel_size = info.output_width * info.output_height;
    int n = location / channel_size;
    int loc = location % channel_size;
    return batch * info.output_number + n * channel_size * (5 + info.classes) + entry * channel_size + loc;
}

vector<Detection> YOLOv3Layer::yolo_get_detection_without_correction() {
    float threshold = 0.1; // TODO: threshold input
    float *feature = output_tensor->weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    
    vector<Detection> dets;

    for (int i = 0; i < channel_size; ++i){
        int row = i / width;
        int col = i % width;
        for(int n = 0; n < info.anchor_num; ++n){
            int obj_index  = entry_index(0, n * channel_size + i, 4);
            float objectness = feature[obj_index];
            if(objectness <= threshold)
                continue;
            Detection det; det.prob.resize(classes);
            int box_index  = entry_index(0, n * channel_size + i, 0);
            det.bbox = get_yolo_box(feature, bias, mask[n], box_index, col, row, width, height, net_width, net_height, channel_size);
            det.objectness = objectness;
            det.classes = info.classes;
            for(int j = 0; j < info.classes; ++j){
                int class_index = entry_index(0, n * channel_size + i, 4 + 1 + j);
                float prob = objectness * feature[class_index];
                det.prob[j] = (prob > threshold) ? prob : 0;
            }
            dets.push_back(det);
        }
    }
    
    return dets;
}

Box YOLOv3Layer::get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    Box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n + 0] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
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

YOLOv4Layer::YOLOv4Layer(LayerOption opt_) {
    opt = opt_;
    type = LayerType::Yolov4;
    name = (opt.find("name") == opt.end()) ? "yolov4" : opt["name"];
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
    info.ignore_iou_threshold = (opt.find("ignore_iou_threshold") == opt.end()) ? 0.5 : atof(opt["ignore_iou_threshold"].c_str());
    info.truth_iou_threshold = (opt.find("truth_iou_threshold") == opt.end()) ? 1 : atof(opt["truth_iou_threshold"].c_str());
    
    info.scale_x_y = (opt.find("scale_x_y") == opt.end()) ? 1 : atof(opt["scale_x_y"].c_str());
    info.iou_normalizer = (opt.find("iou_normalizer") == opt.end()) ? 0.75 : atof(opt["iou_normalizer"].c_str());
    info.obj_normalizer = (opt.find("obj_normalizer") == opt.end()) ? 1 : atof(opt["obj_normalizer"].c_str());
    info.cls_normalizer = (opt.find("cls_normalizer") == opt.end()) ? 1 : atof(opt["cls_normalizer"].c_str());
    info.delta_normalizer = (opt.find("delta_normalizer") == opt.end()) ? 1 : atof(opt["delta_normalizer"].c_str());
    info.beta_nms = (opt.find("beta_nms") == opt.end()) ? 0.6 : atof(opt["beta_nms"].c_str());
    info.yolov4_new_coordinate = (opt.find("new_coordinate") != opt.end());
    
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
    
    biases = new Tensor(2, 1, info.total_anchor_num, 0); // Anchor
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
    detection = Tensor(1, 1, 1, 0);
}

Tensor* YOLOv4Layer::connectGraph(Tensor* input_tensor_, vtensorptr extra_tensor_, float *workspace) {
    input_tensor = input_tensor_;
    return &detection;
}

void YOLOv4Layer::Forward(bool train) {
    float *input = input_tensor->weight;
    float *output = output_tensor->weight;
    int channel_size = info.output_width * info.output_height;
    float scale_x_y = info.scale_x_y;
    float scale_x_y_bias = -0.5 * (scale_x_y - 1);
//    bool new_coordinate = info.yolov4_new_coordinate;
    bool new_coordinate = false;
    
    memcpy(output, input, info.output_number * info.batch_size * sizeof(float));

    for (int b = 0; b < info.batch_size; ++b){
        for(int n = 0; n < info.anchor_num; ++n) {
            int bbox_index = entry_index(b, n * channel_size, 0);
            if (new_coordinate) {
                
            } else {
                activate_array(output + bbox_index, 2 * channel_size, LOGISTIC);
                int obj_index = entry_index(b, n * channel_size, 4);
                activate_array(output + obj_index, (1 + info.classes) * channel_size, LOGISTIC);
            }
            scal_add_cpu(2 * channel_size, scale_x_y, scale_x_y_bias, output + bbox_index);
        }
    }
    cout << *output_tensor;
    input_tensor->clearDeltaWeight();
    
    if (!train) {
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
    }
}

float YOLOv4Layer::Backward(Tensor *target) {
    float ignore_iou_threshold = 0.7; // TODO: should be input
    float truth_iou_threshold = 1;
    
    int width = info.input_width;
    int height = info.input_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int anchor_num = info.anchor_num;
    int total_anchor_num = info.total_anchor_num;
    int channel_size = width * height;
    int max_boxes = info.max_boxes;
    int target_size = 5 * max_boxes;
    int classes = info.classes;
    
    float *output = output_tensor->weight;
    float *delta = input_tensor->delta_weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    float *target_ptr = target->weight;
    
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    int i,j,b,t,n;
    
    float loss = 0;
    for (b = 0; b < info.batch_size; ++b) {
        for (j = 0; j < height; ++j) {
            for (i = 0; i < width; ++i) {
                for (n = 0; n < anchor_num; ++n) {
                    int box_index = entry_index(b, n * channel_size + j * width + i, 0);
                    Box pred = get_yolo_box(output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, channel_size);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < max_boxes; ++t){
                        Box truth = float_to_box(target_ptr + t * (4 + 1) + target_size, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    int obj_index = entry_index(b, n * channel_size + j * width + i, 4);
                    avg_anyobj += output[obj_index];
                    delta[obj_index] = output[obj_index];
                    if (best_iou > ignore_iou_threshold) {
                        delta[obj_index] = 0;
                    }
                    if (best_iou > truth_iou_threshold) {
                        delta[obj_index] = output[obj_index] - 1;

                        int cls = target_ptr[best_t*(4 + 1) + b * target_size + 4];
                        int class_index = entry_index(b, n * channel_size + j * width + i, 4 + 1);
                        delta_yolo_class(output, delta, class_index, cls, classes, channel_size, 0);
                        Box truth = float_to_box(target_ptr + best_t * (4 + 1) + b * target_size, 1);
                        delta_yolo_box(truth, output, bias, mask[n], box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);
                    }
                }
            }
        }
        for(t = 0; t < max_boxes; ++t){
            Box truth = float_to_box(target_ptr + t * (4 + 1) + b * target_size, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * width);
            j = (truth.y * height);
            Box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            for(n = 0; n < total_anchor_num; ++n){
                Box pred = {0};
                pred.w = bias[2 * n + 0] / net_width;
                pred.h = bias[2 * n + 1] / net_height;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int mask_n = int_index(mask, best_n, anchor_num);
            if(mask_n >= 0){
                int box_index = entry_index(b, mask_n * channel_size + j * width + i, 0);
                float iou = delta_yolo_box(truth, output, bias, best_n, box_index, i, j, width, height, net_width, net_height, delta, (2 - truth.w * truth.h), channel_size);

                int obj_index = entry_index(b, mask_n * channel_size + j * width + i, 4);
                avg_obj += output[obj_index];
                delta[obj_index] = output[obj_index] - 1;

                int cls = target_ptr[t * (4 + 1) + b * target_size + 4];
                int class_index = entry_index(b, mask_n * channel_size + j * width + i, 4 + 1);
                delta_yolo_class(output, delta, class_index, cls, classes, channel_size, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    loss = pow(mag_array(delta, info.output_number * info.batch_size), 2);
    printf("Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(channel_size*anchor_num*info.batch_size), recall/count, recall75/count, count);
    return loss;
}

void YOLOv4Layer::delta_yolo_class(float *output, float *delta, int index, int cls, int classes, int stride, float *avg_cat) {
    if (delta[index]) {
        delta[index + stride * cls] = output[index + stride * cls] - 1;
        if(avg_cat)
            *avg_cat += output[index + stride * cls];
        return;
    }
    for(int n = 0; n < classes; ++n) {
        delta[index + stride * n] = output[index + stride * n] - ((n == cls) ? 1 : 0);
        if(n == cls && avg_cat)
            *avg_cat += output[index + stride * n];
    }
}

float YOLOv4Layer::delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride) {
    Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);
    
    float tx = (truth.x * lw - i);
    float ty = (truth.y * lh - j);
    float tw = log(truth.w * w / biases[2 * n + 0]);
    float th = log(truth.h * h / biases[2 * n + 1]);
    
    delta[index + 0 * stride] = -scale * (tx - x[index + 0 * stride]);
    delta[index + 1 * stride] = -scale * (ty - x[index + 1 * stride]);
    delta[index + 2 * stride] = -scale * (tw - x[index + 2 * stride]);
    delta[index + 3 * stride] = -scale * (th - x[index + 3 * stride]);
    return iou;
}

int YOLOv4Layer::entry_index(int batch, int location, int entry) {
    int channel_size = info.output_width * info.output_height;
    int n = location / channel_size;
    int loc = location % channel_size;
    return batch * info.output_number + n * channel_size * (5 + info.classes) + entry * channel_size + loc;
}

vector<Detection> YOLOv4Layer::yolo_get_detection_without_correction() {
    float threshold = 0.01; // TODO: threshold input
    float *feature = output_tensor->weight;
    float *bias = biases->weight;
    float *mask = kernel[0].weight;
    
    int width = info.output_width;
    int height = info.output_height;
    int net_width = info.net_width;
    int net_height = info.net_height;
    int classes = info.classes;
    int channel_size = width * height;
    
    vector<Detection> dets;

    for (int i = 0; i < channel_size; ++i){
        int row = i / width;
        int col = i % width;
        for(int n = 0; n < info.anchor_num; ++n){
            int obj_index  = entry_index(0, n * channel_size + i, 4);
            float objectness = feature[obj_index];
            if(objectness <= threshold)
                continue;
            Detection det; det.prob.resize(classes);
            int box_index  = entry_index(0, n * channel_size + i, 0);
            det.bbox = get_yolo_box(feature, bias, mask[n], box_index, col, row, width, height, net_width, net_height, channel_size);
            det.objectness = objectness;
            det.classes = info.classes;
            for(int j = 0; j < info.classes; ++j){
                int class_index = entry_index(0, n * channel_size + i, 4 + 1 + j);
                float prob = objectness * feature[class_index];
                det.prob[j] = (prob > threshold) ? prob : 0;
            }
            dets.push_back(det);
        }
    }
    
    return dets;
}

Box YOLOv4Layer::get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    Box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n + 0] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

float YOLOv4Layer::mag_array(float *a, int n) {
    float sum = 0;
    for(int i = 0; i < n; ++i){
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

int YOLOv4Layer::int_index(float *a, int val, int n) {
    for(int i = 0; i < n; ++i) {
        if(a[i] == val)
            return i;
    }
    return -1;
}

void add_bias(float *output, float *biases, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size) {
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta + size * (i + b * n), size);
        }
    }
}
