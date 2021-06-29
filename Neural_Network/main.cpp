//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>

#include "Neural_Network.hpp"

using namespace std;

class PNet {
public:
    PNet();
private:
    Neural_Network pnet;
};

int main(int argc, const char * argv[]) {
    Neural_Network nn;
    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
    nn.makeLayer();
//    nn.load("model.bin");
    nn.shape();


    Data bee("bee.npy", 28, 28);
    vtensor data_bee = bee.get(500);
    vector<vfloat> label_bee(500, vfloat(1, 0));
    Data cat("cat.npy", 28, 28);
    vtensor data_cat = cat.get(500);
    vector<vfloat> label_cat(500, vfloat(1, 1));
    Data fish("fish.npy", 28, 28);
    vtensor data_fish = fish.get(500);
    vector<vfloat> label_fish(500, vfloat(1, 2));

    vtensor data_train;
    vector<vfloat> data_label;
    for (int i = 0; i < 300; ++i) {
        data_train.push_back(data_bee[i]);
        data_train.push_back(data_cat[i]);
        data_train.push_back(data_fish[i]);
        data_label.push_back(label_bee[i]);
        data_label.push_back(label_cat[i]);
        data_label.push_back(label_fish[i]);
    }

    vtensor data_valid;
    vector<vfloat> label_valid;
    for (int i = 300; i < 500; ++i) {
        data_valid.push_back(data_bee[i]);
        data_valid.push_back(data_cat[i]);
        data_valid.push_back(data_fish[i]);
        label_valid.push_back(label_bee[i]);
        label_valid.push_back(label_cat[i]);
        label_valid.push_back(label_fish[i]);
    }

    printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);

    nn.train("SVG", 0.001, data_train, data_label, 3);

    printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);

    vfloat out = nn.predict(&data_bee[0]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_bee[327]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_bee[376]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    out = nn.predict(&data_cat[15]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_cat[312]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_cat[305]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    out = nn.predict(&data_fish[98]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_fish[312]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_fish[456]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    nn.save("model.bin");
    

//    Neural_Network pnet("parallel");
//    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
//    pnet.addLayer(LayerOption{{"type", "Relu"}, {"name", "relu_1"}});
//    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
//    pnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "4"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"input_name", "conv_3"}});
//    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "conv_5"}, {"alpha", "0.5"}});
//    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"input_name", "conv_3"}});
//    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "conv_6"}, {"alpha", "0.5"}});
//    pnet.addOutput("cls_prob");
//    pnet.addOutput("bbox_pred");
//    pnet.addOutput("land_pred");
//    pnet.makeLayer();
//    pnet.shape();
//
//    Tensor test(12, 12, 3);
//    vfloat result = pnet.Forward(&test);
//    for (int i = 0; i < result.size(); ++i) {
//        cout << result[i] << endl;
//    }
//    pnet.save("pnet.bin");
    
//    cout << "\n\n\n";
//
//    Neural_Network nnt;
//    nnt.load("pnet.bin");
//    nnt.shape();
//    nnt.Forward(&test);
//    for (int i = 0; i < result.size(); ++i) {
//        cout << result[i] << endl;
//    }
//    PNet pnet;
    
    return 0;
}

PNet::PNet() {
    pnet = Neural_Network("parallel");
    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    pnet.addLayer(LayerOption{{"type", "Relu"}, {"name", "relu_1"}});
    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
    pnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "4"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"input_name", "conv_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "conv_5"}, {"alpha", "0.5"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"input_name", "conv_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "conv_6"}, {"alpha", "0.5"}});
    pnet.addOutput("cls_prob");
    pnet.addOutput("bbox_pred");
    pnet.addOutput("land_pred");
    pnet.makeLayer();
    pnet.shape();
}
