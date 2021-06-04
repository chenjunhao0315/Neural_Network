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
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "1"}, {"input_height", "1"}, {"input_dimension", "2"}});
//    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Relu"}});
//    nn.addLayer(LayerOption{{"type", "Softmax"}, {"number_class", "2"}});
    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "64"}, {"activation", "Relu"}});
    nn.addLayer(LayerOption{{"type", "Softmax"}, {"number_class", "3"}});
    nn.makeLayer();
//    nn.shape();
    
//    Tensor test1(5, 5, 1, 0), test2(5, 5, 1, 1);
//    vtensor test{{Tensor(5, 5, 1, 0)}, {Tensor(5, 5, 1, 0.3)}, {Tensor(5, 5, 1, 0.7)}, Tensor{5, 5, 1, 1}};
//    vfloat label{0, 0, 1, 1};
//    vfloat test_result = nn.predict(&test1);
//    printf("Predict: %.0f (%.2f%%)\n", test_result[0], test_result[1] * 100);
//    test_result = nn.predict(&test2);
//    printf("Predict: %.0f (%.2f%%)\n", test_result[0], test_result[1] * 100);
////    nn.shape();
//    nn.train("SVG", 0.001, test, label, 300);
////    nn.shape();
//    test_result = nn.predict(&test1);
//    printf("Predict: %.0f (%.2f%%)\n", test_result[0], test_result[1] * 100);
//    test_result = nn.predict(&test2);
//    printf("Predict: %.0f (%.2f%%)\n", test_result[0], test_result[1] * 100);
    
    
//    vtensor data_set{Tensor(vfloat{0, 0}), Tensor(vfloat{0, 1}), Tensor(vfloat{1, 0}), Tensor(vfloat{1, 1})};
//    vfloat data_label{0, 1, 1, 0};
    
//    Tensor input(1, 1, 2, 1);
//    vfloat result = nn.predict(&input);
//    printf("Predict: %.0f (%.2f%%)\n", result[0], result[1] * 100);
    
    // train
//    nn.train("SVG", 0.01, data_set, data_label, 176);
    
//    result = nn.predict(&input);
//    printf("Predict: %.0f (%.2f%%)\n", result[0], result[1] * 100);
    
    Data bee("bee.npy", 28, 28);
    vtensor data_bee = bee.get(500);
    vfloat label_bee(500, 0);
    Data cat("cat.npy", 28, 28);
    vtensor data_cat = cat.get(500);
    vfloat label_cat(500, 1);
    Data fish("fish.npy", 28, 28);
    vtensor data_fish = fish.get(500);
    vfloat label_fish(500, 2);
    vtensor data_train(data_bee);
    vfloat data_label(label_bee);
    for (int i = 0; i < data_cat.size(); ++i) {
        data_train.push_back(data_cat[i]);
        data_label.push_back(label_cat[i]);
        data_train.push_back(data_fish[i]);
        data_label.push_back(label_fish[i]);
    }
//    data_train.insert(data_bee.end(), data_cat.begin(), data_cat.end());
    
//    data_label.insert(label_bee.end(), label_cat.begin(), label_cat.end());
    
    
    vfloat out = nn.predict(&data_bee[0]);
    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    nn.shape();
    nn.train("SVG", 0.01, data_train, data_label, 1);
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
    
//    ConvolutionLayer cl(LayerOption{{"type", "Convoluation"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}});
////    cl.shape();
//    Tensor *out = cl.Forward(&img);
//    printf("%d %d %d\n", out->getWidth(), out->getHeight(),  out->getDimension());
    
    return 0;
}
