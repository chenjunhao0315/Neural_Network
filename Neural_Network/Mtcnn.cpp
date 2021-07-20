//
//  Mtcnn.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/19.
//

#include "Mtcnn.hpp"

PNet::PNet(const char *model_name) {
    min_face_size = 25;
    scale_factor = 0.709;
    threshold[0] = 0.97;
    pnet = Neural_Network("parallel");
    pnet.load(model_name);
}

PNet::PNet() {
    min_face_size = 25;
    scale_factor = 0.709;
    threshold[0] = 0.96;
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
        
        vector<Bbox> list = generate_bbox(map, current_scale, threshold[0]);
        list = nms(list, 0.5);
        bbox_list.insert(bbox_list.end(), list.begin(), list.end());
        
        current_scale *= scale_factor;
        img_resize = img.resize(Size(), current_scale, current_scale);
        current_width = img_resize.width;
        current_height = img_resize.height;
    }
    
    bbox_list = nms(bbox_list, 0.7);
    
    for (int i = 0; i < bbox_list.size(); ++i) {
        int bbox_w = bbox_list[i].x2 - bbox_list[i].x1 + 1;
        int bbox_h = bbox_list[i].y2 - bbox_list[i].y1 + 1;
        bbox_list[i].x1 += bbox_list[i].dx1 * bbox_w;
        bbox_list[i].x2 += bbox_list[i].dx2 * bbox_w;
        bbox_list[i].y1 += bbox_list[i].dy1 * bbox_h;
        bbox_list[i].y2 += bbox_list[i].dy2 * bbox_h;
    }
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

vector<Bbox> nms(vector<Bbox> &Bbox_list, float threshold) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
        return box1.score < box2.score;
    };
    sort(Bbox_list.begin(), Bbox_list.end(), cmpScore);
    
    vector<Bbox> pickedBbox;
    while (Bbox_list.size() > 0) {
        pickedBbox.emplace_back(Bbox_list.back());
        Bbox_list.pop_back();
        for (size_t i = 0; i < Bbox_list.size(); i++) {
            if (iou(pickedBbox.back(), Bbox_list[i]) >= threshold) {
                Bbox_list.erase(Bbox_list.begin() + i);
            }
        }
    }
    return pickedBbox;
}

float iou(Bbox box1, Bbox box2) {
    int area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    int area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    int x11 = max(box1.x1, box2.x1);
    int y11 = max(box1.y1, box2.y1);
    int x22 = min(box1.x2, box2.x2);
    int y22 = min(box1.y2, box2.y2);
    int width = ((x22 - x11 + 1) > 0) ? (x22 - x11 + 1) : 0;
    int height = ((y22 - y11 + 1) > 0) ? (y22 - y11 + 1) : 0;
    float intersection = width * height;
    return intersection / (area1 + area2 - intersection);
}

vector<Bbox> convert_to_square(vector<Bbox> &Bbox_list) {
    vector<Bbox> square_bbox = Bbox_list;
    for (int i = 0; i < square_bbox.size(); ++i) {
        int h = Bbox_list[i].y2 - Bbox_list[i].y1;
        int w = Bbox_list[i].x2 - Bbox_list[i].x1;
        int max_side = (h > w) ? h : w;
        square_bbox[i].x1 = round(Bbox_list[i].x1 + w * 0.5 - max_side * 0.5);
        square_bbox[i].y1 = round(Bbox_list[i].y1 + h * 0.5 - max_side * 0.5);
        square_bbox[i].x2 = round(square_bbox[i].x1 + max_side - 1);
        square_bbox[i].y2 = round(square_bbox[i].y1 + max_side - 1);
    }
    return square_bbox;
}

vector<Bbox> calibrate_box(vector<Bbox> &Bbox_list) {
    vector<Bbox> bbox_c = Bbox_list;
    size_t Bbox_c_size = bbox_c.size();
    for (int i = 0; i < Bbox_c_size; ++i) {
        int width = bbox_c[i].x2 - bbox_c[i].x1;
        int height = bbox_c[i].y2 - bbox_c[i].y1;
        bbox_c[i].x1 += bbox_c[i].dx1 * width;
        bbox_c[i].y1 += bbox_c[i].dy1 * height;
        bbox_c[i].x2 += bbox_c[i].dx2 * width;
        bbox_c[i].y2 += bbox_c[i].dy2 * height;
    }
    return bbox_c;
}

vector<Bbox> RNet::detect(IMG &img, vector<Bbox> &pnet_bbox) {
    vector<Bbox> square_bbox = convert_to_square(pnet_bbox);
    vector<Bbox> rnet_bbox;
    size_t square_bbox_size = square_bbox.size();
    for (int i = 0; i < square_bbox_size; ++i) {
        IMG crop = img.crop(Rect(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2));
        crop = crop.resize(Size(24, 24));
        unsigned char *pixel = crop.toRGB();
        float *pixel_c = new float [3 * 24 * 24];
        for (int i = 0; i < 3 * 24 * 24; ++i) {
            pixel_c[i] = ((float)pixel[i] - 127.5) / 128;
        }
        Tensor crop_img(pixel_c, 24, 24, 3);
        vfloat rnet_detect = rnet.Forward(&crop_img);
        if (rnet_detect[1] > threshold[0]) {
            Bbox rnet_detect_box(square_bbox[i].x1, square_bbox[i].y1, square_bbox[i].x2, square_bbox[i].y2, rnet_detect[1], rnet_detect[2], rnet_detect[3], rnet_detect[4], rnet_detect[5]);
            rnet_bbox.push_back(rnet_detect_box);
        }
    }
    rnet_bbox = nms(rnet_bbox, 0.6);
    rnet_bbox = calibrate_box(rnet_bbox);
    return rnet_bbox;
}

RNet::RNet() {
    threshold[0] = 0.9;
    rnet = Neural_Network("parallel");
    rnet.addLayer(LayerOption{{"type", "Input"}, {"input_width", "24"}, {"input_height", "24"}, {"input_dimension", "3"}, {"name", "input"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "28"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_1"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_1"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_1"}, {"padding", "1"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "48"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_2"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_2"}});
    rnet.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "3"}, {"stride", "2"}, {"name", "pool_2"}});
    rnet.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "2"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}});
    rnet.addLayer(LayerOption{{"type", "PRelu"}, {"name", "prelu_3"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "128"}, {"name", "fc_1"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"name", "fc_cls"}});
    rnet.addLayer(LayerOption{{"type", "Softmax"}, {"name", "cls_prob"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}, {"name", "fc_bbox"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "bbox_pred"}, {"input_name", "fc_bbox"}});
    rnet.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "10"}, {"name", "fc_land"}, {"input_name", "fc_1"}});
    rnet.addLayer(LayerOption {{"type", "EuclideanLoss"}, {"name", "land_pred"}, {"input_name", "fc_land"}});
    rnet.addOutput("cls_prob");
    rnet.addOutput("bbox_pred");
    rnet.addOutput("land_pred");
    rnet.makeLayer();
    rnet.shape();
}

RNet::RNet(const char *model_name) {
    rnet = Neural_Network("parallel");
    rnet.load(model_name);
}

void mtcnn_data_loader(const char *data_path, const char *label_path, vtensor &data_set, vector<vfloat> &label_set, int width, int size) {
    FILE *img = fopen(data_path, "rb");
    FILE *label = fopen(label_path, "rb");
    
    for (int i = 0; i < size; ++i) {
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
        
        unsigned char pixel[3 * width * width];
        fread(pixel, sizeof(unsigned char), 3 * width * width, img);
        
        if (i < 0) {
            unsigned char pixel_R[width * width];
            unsigned char pixel_G[width * width];
            unsigned char pixel_B[width * width];
            for (int i = 0; i < width * width; ++i) {
                pixel_R[i] = pixel[i * 3  +  0];
                pixel_G[i] = pixel[i * 3  +  1];
                pixel_B[i] = pixel[i * 3  +  2];
            }
            
            string file_name = to_string(i);
            FILE *f = fopen(file_name.c_str(), "wb");
            fprintf(f, "P6\n%d %d\n255\n", width, width);
            fwrite(pixel, sizeof(unsigned char), 3 * width * width, f);
            fclose(f);
        }
        float normal_pixel[3 * width * width];
        for (int i = 0; i < 3 * width * width; ++i) {
            normal_pixel[i] = ((float)pixel[i] - 127.5) / 128.0;
        }
        data_set.push_back(Tensor(normal_pixel, width, width, 3));
    }
    fclose(img);
    fclose(label);
}

void mtcnn_evaluate(Neural_Network *nn, vtensor &data_set, vector<vfloat> &label_set) {
    int count = 0;
    int correct = 0;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < data_set.size(); ++i) {
        vfloat out = nn->Forward(&data_set[i]);
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
}
