//
//  Test_All_Layer.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/9/3.
//

#include "Test_All_Layer.hpp"

void test_all_layer() {
    Neural_Network nn;
    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"name", "input"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_1"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "PRelu"}, {"name", "conv_2"}});
    nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_1"}, {"name", "shortcut_1"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    nn.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample"}});
    nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "pr_conv_2"}, {"name", "concat"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "1"}, {"name", "pool_2"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_3"}});
    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
    nn.compile(8);
//    nn.load("test.bin");
    nn.shape();
    
    Data bee("bee.npy", 28, 28);
    vtensor data_bee = bee.get(500);
    vtensor label_bee(500, Tensor(1, 1, 1, 0));
    Data cat("cat.npy", 28, 28);
    vtensor data_cat = cat.get(500);
    vtensor label_cat(500, Tensor(1, 1, 1, 1));
    Data fish("fish.npy", 28, 28);
    vtensor data_fish = fish.get(500);
    vtensor label_fish(500, Tensor(1, 1, 1, 2));
    
    vtensor data_train;
    vtensor data_label;
    for (int i = 0; i < 300; ++i) {
        data_train.push_back(data_bee[i]);
        data_train.push_back(data_cat[i]);
        data_train.push_back(data_fish[i]);
        data_label.push_back(label_bee[i]);
        data_label.push_back(label_cat[i]);
        data_label.push_back(label_fish[i]);
    }
    
    vtensor data_valid;
    vtensor label_valid;
    for (int i = 300; i < 500; ++i) {
        data_valid.push_back(data_bee[i]);
        data_valid.push_back(data_cat[i]);
        data_valid.push_back(data_fish[i]);
        label_valid.push_back(label_bee[i]);
        label_valid.push_back(label_cat[i]);
        label_valid.push_back(label_fish[i]);
    }
    
    
//    printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label) * 100);
    
    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"learning_rate", 0.001}, {"sub_division", 8}});

    Clock c;
    trainer.train_batch(data_train, data_label, 10);
    c.stop_and_show();

    printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label) * 100);
//    nn.save("test.bin");
}
