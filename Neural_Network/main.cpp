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
    ~PNet() {pnet.save("pnet.bin");}
    //private:
    Neural_Network pnet;
};

int main(int argc, const char * argv[]) {
    // This is a good day to learn.
    //    Neural_Network nn("parallel");
    //    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
    //    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
    //    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
    //    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
    //    nn.addOutput("8");
    //    nn.makeLayer();
    //    nn.load("model.bin");
    //    nn.shape();
    //
    //
    //    Data bee("bee.npy", 28, 28);
    //    vtensor data_bee = bee.get(500);
    //    vector<vfloat> label_bee(500, vfloat(1, 0));
    //    Data cat("cat.npy", 28, 28);
    //    vtensor data_cat = cat.get(500);
    //    vector<vfloat> label_cat(500, vfloat(1, 1));
    //    Data fish("fish.npy", 28, 28);
    //    vtensor data_fish = fish.get(500);
    //    vector<vfloat> label_fish(500, vfloat(1, 2));
    //
    //    vtensor data_train;
    //    vector<vfloat> data_label;
    //    for (int i = 0; i < 300; ++i) {
    //        data_train.push_back(data_bee[i]);
    //        data_train.push_back(data_cat[i]);
    //        data_train.push_back(data_fish[i]);
    //        data_label.push_back(label_bee[i]);
    //        data_label.push_back(label_cat[i]);
    //        data_label.push_back(label_fish[i]);
    //    }
    //
    //    vtensor data_valid;
    //    vector<vfloat> label_valid;
    //    for (int i = 300; i < 500; ++i) {
    //        data_valid.push_back(data_bee[i]);
    //        data_valid.push_back(data_cat[i]);
    //        data_valid.push_back(data_fish[i]);
    //        label_valid.push_back(label_bee[i]);
    //        label_valid.push_back(label_cat[i]);
    //        label_valid.push_back(label_fish[i]);
    //    }
    //
    //    printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);
    //
    //    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADADELTA}, {"batch_size", 2}});
    //    trainer.train(data_train, data_label, 1);
    //    //nn.train("SVG", 0.001, data_train, data_label, 1);
    //
    //    printf("Accuracy: %.2f%%\n", nn.evaluate(data_valid, label_valid) * 100);
    //    nn.shape();
    //
    //    vfloat out = nn.predict(&data_bee[0]);
    //    printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
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
    //    //
    //    nn.save("model.bin");
    
    
    FILE *img = fopen("img_data.bin", "rb");
    FILE *label = fopen("label_data.bin", "rb");
    vtensor data_set;
    vector<vfloat> label_set;
    
    for (int i = 0; i < 140; ++i) {
        int cls;
        float bbox[4];
        float landmark[10];
        fread(&cls, sizeof(int), 1, label);
        fread(bbox, sizeof(float), 4, label);
        fread(landmark, sizeof(float), 10, label);
        vfloat label;
        label.push_back((float)cls);
        for (int i = 0; i < 4; ++i) {
            label.push_back(bbox[i]);
        }
        for (int i = 0; i < 10; ++i) {
            label.push_back(landmark[i]);
        }
        label_set.push_back(label);
        //        printf("cls: %d\nbbox: (%.2f, %.2f, %.2f, %.2f)\nlandmark: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)\n", cls, bbox[0], bbox[1], bbox[2], bbox[3], landmark[0], landmark[1], landmark[2], landmark[3], landmark[4], landmark[5], landmark[6], landmark[7], landmark[8], landmark[9]);
        //        printf("cls: %d\nbbox: (%.2f, %.2f, %.2f, %.2f)\nlandmark: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)\n", (int)label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9], label[10], label[11], label[12], label[13], label[14]);
        
        unsigned char pixel[3 * 12 * 12];
        fread(pixel, sizeof(unsigned char), 3 * 12 * 12, img);
        
        if (i < 2) {
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
    
                    PNet pnet("pnet.bin");
//    PNet pnet;
    
    
    vfloat result = pnet.pnet.Forward(&data_set[0]);
    for (int i = 0; i < 2; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    for (int i = 2; i < 6; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    
    
    Trainer trainer(&pnet.pnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 1}, {"l2_decay", 0.0001}});
    trainer.train(data_set, label_set, 100);
    //pnet.pnet.train("SVG", 0.001, data_set, label_set, 1);
    
    result = pnet.pnet.Forward(&data_set[0]);
    cout << "cls: " << label_set[0][0] << endl;
    for (int i = 0; i < 2; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    for (int i = 2; i < 6; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    result = pnet.pnet.Forward(&data_set[1]);
    cout << "cls: " << label_set[1][0] << endl;
    for (int i = 0; i < 2; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    for (int i = 2; i < 6; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    result = pnet.pnet.Forward(&data_set[10]);
    cout << "cls: " << label_set[10][0] << endl;
    for (int i = 0; i < 2; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    for (int i = 2; i < 6; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    //            pnet.pnet.shape();
    
    /*result = pnet.pnet.Forward(&data_set[4001]);
     cout << "cls: " << label_set[4001][0] << endl;
     for (int i = 0; i < 2; ++i) {
     cout << result[i] << endl;
     }
     result = pnet.pnet.Forward(&data_set[5001]);
     cout << "cls: " << label_set[5001][0] << endl;
     for (int i = 0; i < 2; ++i) {
     cout << result[i] << endl;
     }*/
    //        pnet.pnet.shape();
    //    Neural_Network nn;
    //    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "1"}, {"input_height", "1"}, {"input_dimension", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}});
    //    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
    //    nn.makeLayer();
    //    nn.shape();
    //
    //    Tensor test(1, 1, 2, 1);
    //    vfloat label{0};
    //    vfloat result = nn.Forward(&test);
    //    printf("get %f %f\n", result[0], result[1]);
    //    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADADELTA}, {"batch_size", 4}});
    //    for (int i = 0; i < 20; ++i)
    //        trainer.train(test, label);
    //    result = nn.Forward(&test);
    //    printf("get %f %f\n", result[0], result[1]);
    
    //    FILE *cifar_10_1 = fopen("data_batch_1.bin", "rb");
    //    vtensor data;
    //    vector<vfloat> label;
    //    for (int i = 0; i < 400; ++i) {
    //        char cls;
    //        unsigned char red[32 * 32];
    //        unsigned char green[32 * 32];
    //        unsigned char blue[32 * 32];
    //        fread(&cls, 1, 1, cifar_10_1);
    //        fread(red, 1, 32 * 32, cifar_10_1);
    //        fread(green, 1, 32 * 32, cifar_10_1);
    //        fread(blue, 1, 32 * 32, cifar_10_1);
    //        unsigned char pixel[3 * 32 * 32];
    //        for (int j = 0; j < 32 * 32; ++j) {
    //            pixel[j * 3 + 0] = red[j];
    //            pixel[j * 3 + 1] = green[j];
    //            pixel[j * 3 + 2] = blue[j];
    //        }
    //        float normal_pixel[3 * 32 * 32];
    //        for (int i = 0; i < 3 * 32 * 32; ++i) {
    //            normal_pixel[i] = (float)pixel[i] / 255.0;
    //        }
    //        data.push_back(Tensor(normal_pixel, 32, 32, 3));
    //        label.push_back(vfloat{(float)(int)cls});
    //        //        FILE *out = fopen(to_string(i).c_str(), "wb");
    //        //        fprintf(out, "P6\n32 32\n255\n");
    //        //        fwrite(pixel, sizeof(unsigned char), 3 * 32 * 32, out);
    //        //        fclose(out);
    //    }
    //    fclose(cifar_10_1);
    
    //    Neural_Network nn;
    //    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "32"}, {"input_height", "32"}, {"input_dimension", "3"}});
    //    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "2"}, {"activation", "PRelu"}});
    //    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "2"}, {"activation", "PRelu"}});
    //    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "2"}, {"activation", "PRelu"}});
    //    nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
    //    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "10"}, {"activation", "Softmax"}});
    //    nn.makeLayer();
    //    nn.load("cifar-10-n.bin");
    //    nn.shape();
    //    float acc_b;
    //    printf("Accuarcy: %.2f%%\n", (acc_b = nn.evaluate(data, label) * 100));
    //
    //    float acc = 0;
    //    while(acc <= acc_b) {
    //        Neural_Network *nn1 = new Neural_Network;
    //        nn1->load("cifar-10-n.bin");
    //        Trainer trainer1(nn1, TrainerOption{{"method", Trainer::Method::SGD}, {"batch_size", 1}, {"l2_decay", 0.0001}, {"learning_rate", 0.000005}});
    //        trainer1.train(data, label, 1);
    //        printf("Accuarcy: %.2f%%\n", (acc = nn1->evaluate(data, label) * 100));
    //        if (acc > acc_b)
    //            nn1->save("cifar-10-n.bin");
    //        delete nn1;
    //    }
    
    
    //        Neural_Network nn;
    //        nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}});
    //        nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"activation", "PRelu"}});
    //        nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"activation", "PRelu"}});
    //        nn.addLayer(LayerOption{{"type", "Pooling"}, {"stride", "2"}, {"kernel_width", "2"}});
    //        nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"activation", "PRelu"}});
    //        nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
    //        nn.makeLayer();
    //        nn.shape();
    //
    //        vfloat result = nn.Forward(&data_set[0]);
    //        for (int i = 0; i < 2; ++i) {
    //            cout << result[i] << " ";
    //        }
    //        cout << endl;
    //        Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 1}, {"l2_decay", 0.0001}});
    //        trainer.train(data_set, label_set, 100);
    //
    //        result = nn.Forward(&data_set[0]);
    //        for (int i = 0; i < 2; ++i) {
    //            cout << result[i] << " ";
    //        }
    //        cout << endl;
    //        result = nn.Forward(&data_set[1]);
    //        for (int i = 0; i < 2; ++i) {
    //            cout << result[i] << " ";
    //        }
    //        cout << endl;
    //        result = nn.Forward(&data_set[10]);
    //        for (int i = 0; i < 2; ++i) {
    //            cout << result[i] << " ";
    //        }
    //        cout << endl;
    return 0;
}

PNet::PNet(const char *model_name) {
    pnet = Neural_Network("parallel");
    pnet.load(model_name);
}

PNet::PNet() {
    pnet = Neural_Network("parallel");
    pnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "12"}, {"input_height", "12"}, {"input_dimension", "3"}, {"name", "input"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    pnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    pnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    pnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "2"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}});
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
