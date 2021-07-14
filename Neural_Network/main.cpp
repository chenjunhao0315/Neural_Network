//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>

#include "Neural_Network.hpp"
#include "Data_Process.hpp"
#include "jpeg.h"

using namespace std;

class PNet {
public:
    PNet();
    PNet(const char *model_name);
    ~PNet() {
        pnet.save("pnet2.bin");
    }
    //private:
    Neural_Network pnet;
};

int main(int argc, const char * argv[]) {
    // This is a good day to learn.
//        Neural_Network nn;
//                nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
//                nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
//                nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//                nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
//                nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//                nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
//                nn.makeLayer();
//        nn.load("model.bin");
//        nn.shape();
//
//        Data bee("bee.npy", 28, 28);
//        vtensor data_bee = bee.get(500);
//        vector<vfloat> label_bee(500, vfloat(1, 0));
//        Data cat("cat.npy", 28, 28);
//        vtensor data_cat = cat.get(500);
//        vector<vfloat> label_cat(500, vfloat(1, 1));
//        Data fish("fish.npy", 28, 28);
//        vtensor data_fish = fish.get(500);
//        vector<vfloat> label_fish(500, vfloat(1, 2));
//
//        vtensor data_train;
//        vector<vfloat> data_label;
//        for (int i = 0; i < 300; ++i) {
//            data_train.push_back(data_bee[i]);
//            data_train.push_back(data_cat[i]);
//            data_train.push_back(data_fish[i]);
//            data_label.push_back(label_bee[i]);
//            data_label.push_back(label_cat[i]);
//            data_label.push_back(label_fish[i]);
//        }
//
//        vtensor data_valid;
//        vector<vfloat> label_valid;
//        for (int i = 300; i < 500; ++i) {
//            data_valid.push_back(data_bee[i]);
//            data_valid.push_back(data_cat[i]);
//            data_valid.push_back(data_fish[i]);
//            label_valid.push_back(label_bee[i]);
//            label_valid.push_back(label_cat[i]);
//            label_valid.push_back(label_fish[i]);
//        }
//
//        printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);
//
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 1}, {"learning_rate", 0.0001}});
//        trainer.train(data_train, data_label, 10);
//        //        nn.train("SVG", 0.001, data_train, data_label, 1);
//
//        printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);
//
//        vfloat out = nn.predict(&data_bee[0]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_bee[327]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_bee[376]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//        out = nn.predict(&data_cat[15]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_cat[312]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_cat[305]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//        out = nn.predict(&data_fish[98]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_fish[312]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        out = nn.predict(&data_fish[456]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//        //
//        nn.save("model.bin");
    
    
    FILE *img = fopen("img_data.bin", "rb");
    FILE *label = fopen("label_data.bin", "rb");
    vtensor data_set;
    vector<vfloat> label_set;

    for (int i = 0; i < 70000; ++i) {
        int cls;
        float bbox[4];
        float landmark[10];
        fread(&cls, sizeof(int), 1, label);
        fread(bbox, sizeof(float), 4, label);
        fread(landmark, sizeof(float), 10, label);
        vfloat label_data;
        label_data.push_back((float)cls);
        for (int i = 0; i < 4; ++i) {
            label_data.push_back(bbox[i]);
        }
        for (int i = 0; i < 10; ++i) {
            label_data.push_back(landmark[i]);
        }
        label_set.push_back(label_data);

        unsigned char pixel[3 * 12 * 12];
        fread(pixel, sizeof(unsigned char), 3 * 12 * 12, img);

        if (i < 0) {
            unsigned char pixel_R[12 * 12];
            unsigned char pixel_G[12 * 12];
            unsigned char pixel_B[12 * 12];
            for (int i = 0; i < 12 * 12; ++i) {
                pixel_R[i] = pixel[i * 3  +  0];
                pixel_G[i] = pixel[i * 3  +  1];
                pixel_B[i] = pixel[i * 3  +  2];
            }

            string file_name = to_string(i);
            FILE *f = fopen(file_name.c_str(), "wb");
            fprintf(f, "P6\n12 12\n255\n");
            fwrite(pixel, sizeof(unsigned char), 3 * 12 * 12, f);
            fclose(f);
        }
        float normal_pixel[3 * 12 * 12];
        for (int i = 0; i < 3 * 12 * 12; ++i) {
            normal_pixel[i] = (float)pixel[i] / 255.0;
        }
        data_set.push_back(Tensor(normal_pixel, 12, 12, 3));
    }
    fclose(img);
    fclose(label);

    PNet pnet("pnet2.bin");

    int count = 0;
    int correct = 0;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < data_set.size(); ++i) {
        vfloat out = pnet.pnet.Forward(&data_set[i]);
        if (label_set[i][0] == 1) {
            if (out[1] > out[0]) {
                correct++;
                pos++;
            }
            count++;
        } else if (label_set[i][0] == 0) {
            if (out[0] > out[1]) {
                correct++;
                neg++;
            }
            count++;
        }
    }
    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
    Trainer trainer(&pnet.pnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"eps", 1e-8}, {"batch_size", 2}, {"l2_decay", 0.0001}, {"learning_rate", 0.0005}});
    trainer.train(data_set, label_set, 10);

    count = 0;
    correct = 0;
    pos = 0;
    neg = 0;
    for (int i = 0; i < data_set.size(); ++i) {
        vfloat out = pnet.pnet.Forward(&data_set[i]);
        if (label_set[i][0] == 1) {
            if (out[1] > out[0]) {
                correct++;
                pos++;
            }
            count++;
        } else if (label_set[i][0] == 0) {
            if (out[0] > out[1]) {
                correct++;
                neg++;
            }
            count++;
        }
    }
    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
    vfloat out = pnet.pnet.Forward(&data_set[9699]);
    printf("cls: %.2f %.2f\nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
//    pnet.pnet.shape();
    
//    Neural_Network nn;
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"activation", "Relu"}});
//    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"activation", "Relu"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"activation", "PRelu"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"activation", "Softmax"}});
////    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
//    nn.makeLayer();
////    nn.load("clsadamrevise.bin");
//    nn.shape();
//    cout << "Acc: " << (nn.evaluate(data_set, label_set)) * 100 << "%\n";
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 4}, {"learning_rate", 0.003}});
//    trainer.train(data_set, label_set, 3);
//    cout << "Acc: " << (nn.evaluate(data_set, label_set)) * 100 << "%\n";
//    nn.save("clsadamrevise.bin");

//    for (int i = 0; i < 10000; ++i) {
//        vfloat out = nn.predict(&data_set[i]);
//        printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    }
//    data_set[0].toIMG("1.ppm");
    
//    Neural_Network nn;
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "3"}, {"input_height", "3"}, {"input_dimension", "3"}});
//    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "3"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}});
//    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
//    nn.makeLayer();
//    nn.shape();
//    vfloat one(9, 1), zero(9, 0);
//    Tensor R(one, zero, zero, 3, 3), G(zero, one, one, 3, 3), B(zero, zero, one, 3, 3);
//    vtensor data_set{R, G, B};
//    vector<vfloat> label_set{vfloat{0}, vfloat{1}, vfloat{2}};
////    cout << "Acc: " << (nn.evaluate(data_set, label_set) * 100) << "%\n";
//    vfloat out = nn.predict(&data_set[0]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_set[1]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_set[2]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//    Trainer trainer{&nn, TrainerOption{{"method", Trainer::Method::ADAM}}};
//    trainer.train(data_set, label_set, 1000);
//
//    out = nn.predict(&data_set[0]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_set[1]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    out = nn.predict(&data_set[2]);
//    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//    nn.shape();
    
//    Neural_Network nn;
//    nn.load("pnet2.bin");
//    nn.shape();
//    vfloat out = nn.Forward(&data_set[15001]);
//    printf("cls: %.2f %.2f\nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
//    data_set[19999].toIMG("test.ppm");
    
    
    return 0;
}

PNet::PNet(const char *model_name) {
    pnet = Neural_Network("parallel");
    pnet.load(model_name);
}

PNet::PNet() {
    pnet = Neural_Network("parallel");
    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
        pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
//    pnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}});
    pnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "4"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "conv_5"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "10"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"input_name", "prelu_3"}});
    pnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "conv_6"}});
    pnet.addOutput("cls_prob");
    pnet.addOutput("bbox_pred");
    pnet.addOutput("land_pred");
    pnet.makeLayer();
    pnet.shape();
}
