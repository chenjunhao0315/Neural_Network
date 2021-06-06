//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>

#include "Neural_Network.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    Neural_Network nn;
    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
//    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "64"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Softmax"}, {"number_class", "3"}});
    nn.makeLayer();
    nn.shape();

    
    Data bee("bee.npy", 28, 28);
    vtensor data_bee = bee.get(500);
    vfloat label_bee(500, 0);
    Data cat("cat.npy", 28, 28);
    vtensor data_cat = cat.get(500);
    vfloat label_cat(500, 1);
    Data fish("fish.npy", 28, 28);
    vtensor data_fish = fish.get(500);
    vfloat label_fish(500, 2);
    vtensor data_train;
    data_train.resize(900);
    vfloat data_label;
    data_label.resize(900);
    for (int i = 0; i < 300; ++i) {
        data_train[i * 3] = data_bee[i];
        data_label[i * 3] = label_bee[i];
        data_train[i * 3 + 1] = data_cat[i];
        data_label[i * 3 + 1] = label_cat[i];
        data_train[i * 3 + 2] = data_fish[i];
        data_label[i * 3 + 2] = label_fish[i];
    }

//    printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label));
//    nn.shape();
    nn.train("SVG", 0.01, data_train, data_label, 1);
//    nn.shape();

    vfloat out = nn.predict(&data_bee[0]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_bee[327]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_bee[376]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//    out = nn.predict(&data_cat[15]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_cat[312]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_cat[305]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//    out = nn.predict(&data_fish[98]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_fish[312]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_fish[456]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    
//    FullyConnectedLayer fc(LayerOption{{"type", "Fullyconnected"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"number_neurons", "3"}});
//    fc.save();
    
    return 0;
}
