//
//  Test_All_Layer.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/9/3.
//

#include "Test_All_Layer.hpp"

void test_all_layer(bool save) {
    Neural_Network nn;
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"name", "input"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_1"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "PRelu"}, {"name", "conv_2"}});
//    nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_1"}, {"name", "shortcut_1"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_3"}});
//    nn.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_4"}});
//    nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "pr_conv_2, upsample"}, {"splits", "2"}, {"split_id", "1"}, {"name", "concat"}});
//    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "1"}, {"name", "pool_2"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_5"}});
//    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
//    nn.compile(8);
//    nn.shape();
//    nn.to_prototxt("test_all_layer.prototxt");
    
    nn.addLayer(LayerOption{{"type", "Data"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}, {"scale", "0.0078125"}, {"mean", "127.5, 127.5, 127.5"}, {"name", "input"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Relu"}, {"name", "conv_1"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_2"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_3"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"name", "conv_4"}});
//    nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_2"}, {"name", "shortcut_1"}});
    nn.addLayer(LayerOption{{"type", "Eltwise"}, {"eltwise", "lr_conv_2"}, {"eltwise_op", "prod"}, {"name", "eltwise_1"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_5"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "8"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"name", "conv_6"}, {"input_name", "re_conv_1"}});
    nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_5"}, {"splits", "1"}, {"split_id", "0"}, {"name", "concat"}});
    nn.addLayer(LayerOption{{"type", "Concat"}, {"splits", "2"}, {"split_id", "1"}, {"name", "concat_4"}});
    nn.addLayer(LayerOption{{"type", "Dropout"}, {"probability", "0.2"}, {"name", "dropout"}});
    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"groups", "16"}, {"batchnorm", "true"}, {"activation", "Swish"}, {"name", "conv_7"}, {"input_name", "concat"}});
    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}, {"input_name", "concat"}});
    nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sw_conv_7"}, {"name", "shortcut_2"}});
    nn.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample"}});
    nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "dropout"}, {"splits", "1"}, {"split_id", "0"}, {"name", "concat_3"}});
    nn.addLayer(LayerOption{{"type", "AvgPooling"}, {"name", "avg_pooling"}, {"input_name", "concat"}});
    nn.addLayer(LayerOption{{"type", "ScaleChannel"}, {"scalechannel", "upsample"}, {"name", "scalechannel"}});
    nn.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "avg_pooling"}, {"name", "shortcut_3"}, {"input_name", "concat_3"}});
    nn.addLayer(LayerOption{{"type", "Concat"}, {"concat", "scalechannel, concat_3"}, {"splits", "1"}, {"split_id", "0"}, {"name", "concat_5"}});
    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "32"}, {"name", "connected"}, {"activation", "PRelu"}});
    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "3"}, {"name", "connected_2"}});
    nn.addLayer(LayerOption{{"type", "Softmax"}, {"name", "softmax"}});
    nn.compile(8);
    nn.shape();
//    nn.show_detail();
    nn.to_prototxt("test_all_layer.prototxt");
//    exit(1);
    
    class Data bee("bee.npy", 28, 28, 3, 80);
    vtensor data_bee = bee.get(500);
    vtensor label_bee(500, Tensor(1, 1, 1, 0));
    class Data cat("cat.npy", 28, 28, 3, 80);
    vtensor data_cat = cat.get(500);
    vtensor label_cat(500, Tensor(1, 1, 1, 1));
    class Data fish("fish.npy", 28, 28, 3, 80);
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
    
    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.001}, {"sub_division", 2}, {"warmup", 40}, {"steps", 1}, {"steps_1", 200}, {"scale_1", 0.5}});

    Clock c;
    trainer.train_batch(data_train, data_label, 10);
    c.stop_and_show();

    printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);
//    printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label) * 100);
    if (save) {
//        nn.save("test.bin");
        nn.save_otter("test_all_layer.otter");
        nn.save_ottermodel("test_all_layer.ottermodel");
        printf("Save model finish!\n");
        Neural_Network test;
//        test.load("test.bin");
//        test.load_otter("test_all_layer.otter", "test_all_layer.dam");
        test.load_ottermodel("test_all_layer.ottermodel");
        test.shape();
        test.to_prototxt("test.prototxt");
//        Trainer trainer_test(&test, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.001}, {"sub_division", 2}, {"warmup", 40}, {"steps", 1}, {"steps_1", 200}, {"scale_1", 0.5}});
//        trainer_test.train_batch(data_train, data_label, 1);
        printf("Accuracy: %.2f%%\n", test.evaluate(data_valid, label_valid) * 100);
    }
    
    
}
