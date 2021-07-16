//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>

#include "Neural_Network.hpp"
#include "Data_Process.hpp"
#include "Image_Process.hpp"

using namespace std;

typedef vector<vector<vfloat>> Feature_map;
typedef vector<float> Box;

struct Bbox {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    float dx1;
    float dy1;
    float dx2;
    float dy2;
    Bbox(int x1_, int y1_, int x2_, int y2_, float s, float dx1_, float dy1_, float dx2_, float dy2_): x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s), dx1(dx1_), dy1(dy1_), dx2(dx2_), dy2(dy2_) {};
};

class PNet {
public:
    PNet();
    PNet(const char *model_name);
    ~PNet() {
        pnet.save("pnet_train.bin");
    }
    vector<Bbox> detect(IMG &img);
    Feature_map predict(IMG &img);
    //private:
    Neural_Network pnet;
    int min_face_size;
    float scale_factor;
};

vector<Bbox> generate_bbox(Feature_map map, float scale, float threshold);
vector<Bbox> nms(vector<Bbox> &vecBbox, float threshold);
float iou(Bbox box1, Bbox box2);

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
////        nn.load("model.bin");
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
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 4}});
//        trainer.train(data_train, data_label, 1);
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
        //
//        nn.save("model.bin");
    
    
//    FILE *img = fopen("img_data.bin", "rb");
//    FILE *label = fopen("label_data.bin", "rb");
//    vtensor data_set;
//    vector<vfloat> label_set;
//
//    for (int i = 0; i < 100000; ++i) {
//        int cls;
//        float bbox[4];
//        float landmark[10];
//        fread(&cls, sizeof(int), 1, label);
//        fread(bbox, sizeof(float), 4, label);
//        fread(landmark, sizeof(float), 10, label);
//        vfloat label_data;
//        label_data.push_back((float)cls);
//        for (int i = 0; i < 4; ++i) {
//            label_data.push_back(bbox[i]);
//        }
//        for (int i = 0; i < 10; ++i) {
//            label_data.push_back(landmark[i]);
//        }
//        label_set.push_back(label_data);
//
//        unsigned char pixel[3 * 12 * 12];
//        fread(pixel, sizeof(unsigned char), 3 * 12 * 12, img);
//
//        if (i < 0) {
//            unsigned char pixel_R[12 * 12];
//            unsigned char pixel_G[12 * 12];
//            unsigned char pixel_B[12 * 12];
//            for (int i = 0; i < 12 * 12; ++i) {
//                pixel_R[i] = pixel[i * 3  +  0];
//                pixel_G[i] = pixel[i * 3  +  1];
//                pixel_B[i] = pixel[i * 3  +  2];
//            }
//
//            string file_name = to_string(i);
//            FILE *f = fopen(file_name.c_str(), "wb");
//            fprintf(f, "P6\n12 12\n255\n");
//            fwrite(pixel, sizeof(unsigned char), 3 * 12 * 12, f);
//            fclose(f);
//        }
//        float normal_pixel[3 * 12 * 12];
//        for (int i = 0; i < 3 * 12 * 12; ++i) {
//            normal_pixel[i] = ((float)pixel[i] - 127.5) / 128.0;
//        }
//        data_set.push_back(Tensor(normal_pixel, 12, 12, 3));
//    }
//    fclose(img);
//    fclose(label);
//
//    PNet pnet;
//
//    int count = 0;
//    int correct = 0;
//    int pos = 0;
//    int neg = 0;
////    for (int i = 0; i < data_set.size(); ++i) {
////        vfloat out = pnet.pnet.Forward(&data_set[i]);
////        if (label_set[i][0] == 1) {
////            if (out[1] > out[0]) {
////                correct++;
////                pos++;
////            }
////            count++;
////        } else if (label_set[i][0] == 0) {
////            if (out[0] > out[1]) {
////                correct++;
////                neg++;
////            }
////            count++;
////        }
////    }
////    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
//    Trainer trainer(&pnet.pnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"eps", 1e-14}, {"batch_size", 384}, {"learning_rate", 0.0005}});
//    trainer.train(data_set, label_set, 15);
//
//    count = 0;
//    correct = 0;
//    pos = 0;
//    neg = 0;
//    for (int i = 0; i < data_set.size(); ++i) {
//        vfloat out = pnet.pnet.Forward(&data_set[i]);
//        if (label_set[i][0] == 1) {
//            if (out[1] > out[0]) {
//                correct++;
//                pos++;
//            }
//            count++;
//        } else if (label_set[i][0] == 0) {
//            if (out[0] > out[1]) {
//                correct++;
//                neg++;
//            }
//            count++;
//        }
//    }
//    printf("Acc: %.2f%% pos: %d neg: %d count: %d\n", (float)correct / count * 100, pos, neg, count);
//    vfloat out = pnet.pnet.Forward(&data_set[0]);
//    printf("Predict cls: %.2f %.2f\nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
//    out = label_set[0];
//    printf("Label cls: %.2f \nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14]);
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
//    Trainer trainer{&nn, TrainerOption{{"method", Trainer::Method::ADADELTA}}};
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
    
    PNet pnet("pnet_9393_250k.bin");
    IMG img("7_Cheering_Cheering_7_147.jpg");

    pnet.detect(img);
    
    return 0;
}

PNet::PNet(const char *model_name) {
    min_face_size = 30;
    scale_factor = 0.709;
    pnet = Neural_Network("parallel");
    pnet.load(model_name);
}

PNet::PNet() {
    min_face_size = 30;
    scale_factor = 0.709;
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

vector<Bbox> PNet::detect(IMG &img) {
    int width = img.width;
    int height = img.height;
    int net_size = 12;
    float current_scale = float(net_size) / min_face_size;
    IMG img_resize = img.resize(Size(), current_scale, current_scale);
    int current_width = img_resize.width;
    int current_height = img_resize.height;
    vector<Bbox> bbox_list;
    
    while(min(current_width, current_height) > net_size) {
        Feature_map map = this->predict(img_resize);
        
//        int h = (int)map.size(), w = (int)map[0].size();
//        vfloat show;
//        for (int i = 0; i < h; ++i) {
//            for (int j = 0; j < w; ++j) {
//                if (map[i][j][1] > 0.6)
//                    show.push_back(1);
//                else
//                    show.push_back(0);
//            }
//        }
//        Tensor i(show, show, show, w, h);
//        i.toIMG(to_string(current_scale).c_str());
//        img_resize.convertPPM(to_string(current_width).c_str());
        
        vector<Bbox> list = generate_bbox(map, current_scale, 0.6);
        list = nms(list, 0.5);
        bbox_list.insert(bbox_list.end(), list.begin(), list.end());
        
        current_scale *= scale_factor;
        img_resize = img.resize(Size(), current_scale, current_scale);
        current_width = img_resize.width;
        current_height = img_resize.height;
    }
    IMG pnet_before_nms(img);
    for (int i = 0; i < bbox_list.size(); ++i) {
        pnet_before_nms.drawRectangle(Rect{(bbox_list[i].x1), (bbox_list[i].y1), (bbox_list[i].x2), (bbox_list[i].y2)}, Color{255, 0, 0});
    }
    pnet_before_nms.convertPPM("before_nms.ppm");
    
    bbox_list = nms(bbox_list, 0.7);
    
    IMG pnet_after_nms(img);
    for (int i = 0; i < bbox_list.size(); ++i) {
        pnet_after_nms.drawRectangle(Rect{(bbox_list[i].x1), (bbox_list[i].y1), (bbox_list[i].x2), (bbox_list[i].y2)}, Color{255, 0, 0});
    }
    pnet_after_nms.convertPPM("after_nms.ppm");
    
    for (int i = 0; i < bbox_list.size(); ++i) {
        int bbox_w = bbox_list[i].x2 - bbox_list[i].x1 + 1;
        int bbox_h = bbox_list[i].y2 - bbox_list[i].y1 + 1;
        bbox_list[i].x1 += bbox_list[i].dx1 * bbox_w;
        bbox_list[i].x2 += bbox_list[i].dx2 * bbox_w;
        bbox_list[i].y1 += bbox_list[i].dy1 * bbox_h;
        bbox_list[i].y2 += bbox_list[i].dy2 * bbox_h;
    }
    
    IMG pnet_predict(img);
    
    for (int i = 0; i < bbox_list.size(); ++i) {
        pnet_predict.drawRectangle(Rect{(bbox_list[i].x1), (bbox_list[i].y1), (bbox_list[i].x2), (bbox_list[i].y2)}, Color{255, 0, 0});
    }
    pnet_predict.convertPPM("pnet_predict.ppm");
    
    return bbox_list;
}

Feature_map PNet::predict(IMG &img) {
    int width = img.width;
    int height = img.height;
    int stride = 2;
    int padding = 0;
    int map_width = (width + padding * 2 - 12) / stride + 1;
    int map_height = (height + padding * 2 - 12) / stride + 1;
    Feature_map map(map_height, vector<vfloat>(map_width, vfloat()));
    
    unsigned char *rgb = img.toRGB();
    float *data = new float [3 * 12 * 12];
//    printf("map_width: %d map_height: %d\n", map_width, map_height);
    
    int y = -padding;
    for (int h = 0; h < map_height; ++h, y += stride) {
        int x = -padding;
        for (int w = 0; w < map_width; ++w, x += stride) {
            for (int f_h = 0; f_h < 12; ++f_h) {
                int coordinate_h = y + f_h;
                for (int f_w = 0; f_w < 12; ++f_w) {
                    int coordinate_w = x + f_w;
                    for (int f_d = 0; f_d < 3; ++f_d) {
//                        printf("data: (%d %d %d) src: (%d %d %d)\n", f_w, f_h, f_d, coordinate_w, coordinate_h, f_d);
                        if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height) {
                            data[((12 * f_h) + f_w) * 3 + f_d] = ((float)(rgb[((width * coordinate_h) + coordinate_w) * 3 + f_d] - 127.5) / 128.0);
                        } else {
                            data[((12 * f_h) + f_w) * 3 + f_d] = 0;
                        }
                    }
                }
            }
            Tensor input(data, 12, 12, 3);
//            input.toIMG(to_string(w).c_str());
            vfloat out = pnet.Forward(&input);
            map[h][w] = out;
        }
    }
    delete [] data;
    return map;
}

vector<Bbox> generate_bbox(Feature_map map, float scale, float threshold) {
    int stride = 2;
    int cellsize = 12;
    int height = (int)map.size();
    int width = (int)map[0].size();
    vector<Bbox> bbox_list;
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (map[i][j][1] > threshold) {
                int x1 = (int)round(stride * j / scale);
                int y1 = (int)round(stride * i / scale);
                int x2 = (int)round((stride * j + cellsize) / scale);
                int y2 = (int)round((stride * i + cellsize) / scale);
                float dx1 = map[i][j][2];
                float dy1 = map[i][j][3];
                float dx2 = map[i][j][4];
                float dy2 = map[i][j][5];
                Bbox bbox(x1, y1, x2, y2, map[i][j][1], dx1, dy1, dx2, dy2);
                bbox_list.push_back(bbox);
            }
        }
    }
    return bbox_list;
}

vector<Bbox> nms(vector<Bbox> &vecBbox, float threshold) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
        return box1.score < box2.score;
    };
    sort(vecBbox.begin(), vecBbox.end(), cmpScore);

    vector<Bbox> pickedBbox;
    while (vecBbox.size() > 0) {
        pickedBbox.emplace_back(vecBbox.back());
        vecBbox.pop_back();
        for (size_t i = 0; i < vecBbox.size(); i++) {
            if (vecBbox[i].score == 0)
                continue;
            if (iou(pickedBbox.back(), vecBbox[i]) >= threshold) {
                vecBbox.erase(vecBbox.begin() + i);
            }
        }
    }
    return pickedBbox;
}

//vector<Bbox> nms(vector<Bbox> &vecBbox, float threshold) {
//    auto cmpScore = [](Bbox box1, Bbox box2) {
//        return box1.score > box2.score;
//    };
//    sort(vecBbox.begin(), vecBbox.end(), cmpScore);
//
//    vector<Bbox> pickedBbox;
//    for (int i = 0; i < vecBbox.size(); ++i) {
//        pickedBbox.push_back(vecBbox[i]);
//        for (int j = i + 1; j < vecBbox.size(); ++j) {
//            if (iou(vecBbox[i], vecBbox[j]) >= threshold) {
//                vecBbox.erase(vecBbox.begin() + j);
//                j--;
//            }
//        }
//    }
//    return pickedBbox;
//}

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    int x11 = std::max(box1.x1, box2.x1);
    int y11 = std::max(box1.y1, box2.y1);
    int x22 = std::min(box1.x2, box2.x2);
    int y22 = std::min(box1.y2, box2.y2);
    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);
    return intersection / (area1 + area2 - intersection);
}
