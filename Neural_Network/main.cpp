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
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "128"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Softmax"}, {"number_class", "3"}});
    nn.makeLayer();
    
    Data bee("bee.npy", 28, 28);
    vtensor data_bee = bee.get(100);
    vfloat label_bee(100, 0);
    Data cat("cat.npy", 28, 28);
    vtensor data_cat = cat.get(100);
    vfloat label_cat(100, 1);
    Data fish("fish.npy", 28, 28);
    vtensor data_fish = fish.get(100);
    vfloat label_fish(100, 2);
    vtensor data_train(data_bee);
    vfloat data_label(label_bee);
    for (int i = 0; i < data_cat.size(); ++i) {
        data_train.push_back(data_cat[i]);
        data_label.push_back(label_cat[i]);
        data_train.push_back(data_fish[i]);
        data_label.push_back(label_fish[i]);
    }

    vfloat out = nn.predict(&data_bee[0]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    nn.shape();
    nn.train("SVG", 0.01, data_train, data_label, 3);
//    nn.shape();
    out = nn.predict(&data_bee[0]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_bee[27]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_bee[376]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    out = nn.predict(&data_cat[98]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_cat[312]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_cat[5]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);

    out = nn.predict(&data_fish[198]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_fish[12]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    out = nn.predict(&data_fish[456]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    
    return 0;
}
