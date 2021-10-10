//
//  YOLOv4.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/9/21.
//

#include "YOLOv4.hpp"

YOLOv4::YOLOv4(const char *model_name, int classes_, int batch_size) {
    network = Neural_Network("yolov4");
    network.load(model_name, batch_size);
//    network.load_ottermodel(model_name);
    network.shape();
    classes = classes_;
    label = get_yolo_label("labelstr.txt", classes);
    net_width = 416;
    net_height = 416;
    threshold = 0.45;
}

vector<Detection> YOLOv4::detect(IMG &input) {
    IMG src_img = yolo_pre_process_img(input, net_width, net_height);
    Tensor src_tensor(net_width, net_height, 3, 0);
    convert_index_base_to_channel_base((float *)src_img.toPixelArray(), src_tensor.weight, net_width, net_height, 3);
    
    Clock c;
    vtensorptr feature_map = network.Forward(&src_tensor);
    c.stop_and_show();
    
    vector<Detection> dets;
    for (int i = 0; i < feature_map.size(); ++i) {
        vector<Detection> det = yolo_correct_box(feature_map[i], input.width, input.height, net_width, net_height, 0);
        dets.insert(dets.end(), det.begin(), det.end());
    }
    
    yolo_nms(dets, classes, threshold);
    yolov4_mark(dets, input, 80, 0.24, label);
    
    return vector<Detection>();
}

void yolov4_mark(vector<Detection> &dets, IMG &img, int classes, float threshold, vector<string> label) {
    for (int i = 0; i < dets.size(); ++i) {
        char labelstr[1024] = {0};
        int sort_class = -1;
        for (int c = 0; c < classes; ++c) {
            if (dets[i].prob[c] > threshold) {
                if (sort_class < 0) {
                    strcat(labelstr, label[c].c_str());
                    sort_class = c;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, label[c].c_str());
                }
                printf("%s: %.0f%%\n", label[c].c_str(), dets[i].prob[c]*100);
            }
        }
        if (sort_class >= 0 && dets[i].objectness > 0) {
            IMG c(1, 1, MAT_8UC3, Scalar(255, 0, 0));
            c = c.hsv_distort(1.0 * sort_class / classes, 1, 1);
            unsigned char *color = c.toPixelArray();
            unsigned char r, g, b;
            r = color[0]; g = color[1]; b = color[2];
            int x1 = dets[i].bbox.x - dets[i].bbox.w / 2;
            int y1 = dets[i].bbox.y - dets[i].bbox.h / 2;
            int x2 = dets[i].bbox.x + dets[i].bbox.w / 2;
            int y2 = dets[i].bbox.y + dets[i].bbox.h / 2;
            
            int h = img.height * 0.02;
            if (h < 22)
                h = 22;
            IMG text = textLabel(labelstr, h, BLACK, Color(r, g, b), h * 0.25);
            h = text.height;
            
            img.drawRectangle(Rect(x1, y1, x2, y2), Color(r, g, b));
            img.paste(text, Point(x1, ((y1 - h) < 0) ? y1 : y1 - h));
        }
    }
}

void YOLOv4::yolo_nms(vector<Detection> &det_list, int classes ,float threshold) {
    int i, j, k;
    k = (int)det_list.size() - 1;
    for(i = 0; i <= k; ++i){
        if(det_list[i].objectness == 0){
            Detection swap = det_list[i];
            det_list[i] = det_list[k];
            det_list[k] = swap;
            --k;
            --i;
        }
    }
    int total = k + 1;
    
    auto cmpScore = [](Detection &a, Detection &b) {
        float diff = 0;
        if(b.sort_class >= 0){
            diff = a.prob[b.sort_class] - b.prob[b.sort_class];
        } else {
            diff = a.objectness - b.objectness;
        }
        if(diff < 0) return 1;
        else if(diff > 0) return -1;
        return 0;
    };
    
    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            det_list[i].sort_class = k;
        }
        sort(det_list.begin(), det_list.end(), cmpScore);
        for(i = 0; i < total; ++i){
            if(det_list[i].prob[k] == 0) continue;
            Box a = det_list[i].bbox;
            for(j = i + 1; j < total; ++j){
                Box b = det_list[j].bbox;
                if (box_iou(a, b) > threshold){
                    det_list[j].prob[k] = 0;
                }
            }
        }
    }
}

vector<Detection> YOLOv4::yolo_correct_box(Tensor *box_list, int img_w, int img_h, int net_w, int net_h, bool relative) {
    int list_length = box_list->dimension;
    vector<Detection> corrected_box; corrected_box.resize(list_length);
    float *list_info = box_list->weight;

    int new_w = int(img_w * min((float)net_w / img_w, (float)net_h / img_h));
    int new_h = int(img_h * min((float)net_w / img_w, (float)net_h / img_h));

    for (int i = 0; i < list_length; ++i){
        Detection &det = corrected_box[i];
        Box &bbox = det.bbox;
        bbox.x = (*(list_info++) - ((net_w - new_w) / 2.0 / net_w)) / ((float)new_w / net_w);
        bbox.y = (*(list_info++) - ((net_h - new_h) / 2.0 / net_h)) / ((float)new_h / net_h);
        bbox.w = *(list_info++) * (float)net_w / new_w;
        bbox.h = *(list_info++) * (float)net_h / new_h;
        if(!relative){
            bbox.x *= img_w;
            bbox.w *= img_w;
            bbox.y *= img_h;
            bbox.h *= img_h;
        }
        det.objectness = *(list_info++);
        det.prob.resize(classes);
        for (int c = 0; c < classes; ++c) {
            det.prob[c] = (*(list_info++));
        }
    }
    return corrected_box;
}

IMG YOLOv4::yolo_pre_process_img(IMG &img, int net_w, int net_h) {
    IMG canvas(net_w, net_h, MAT_8UC3, Scalar(128, 128, 128));
    
    int img_w  = img.width, img_h = img.height;
    int new_w = int(img_w * min((float)net_w / img_w, (float)net_h / img_h));
    int new_h = int(img_h * min((float)net_w / img_w, (float)net_h / img_h));
    
    IMG resize = img.resize(Size(new_w, new_h));
    canvas.paste(resize, Point((net_w - new_w) / 2, (net_h - new_h) / 2));
    
    canvas.save("look.jpg");
    canvas = canvas.scale(0.00392157, MAT_32FC3);
    return canvas;
}

vector<string> YOLOv4::get_yolo_label(const char *labelstr, int classes) {
    ifstream f;
    f.open(labelstr);
    if (!f)
        return vector<string>();
    vector<string> label; label.reserve(classes);
    for (int i = 0; i < classes; ++i) {
        string str;
        f >> str;
        label.push_back(str);
    }
    return label;
}

YOLOv4_DataLoader::~YOLOv4_DataLoader() {
    dataset.clear();
}

YOLOv4_DataLoader::YOLOv4_DataLoader(const char *filename) {
    ifstream train_data;
    train_data.open(filename);
    const int max_obj_img = 4000;   // 30000;
    
    while(!train_data.eof()) {
        string filename;
        train_data >> filename;
        int box_num;
        train_data >> box_num;
        yolo_label_v4 label; label.boxes.reserve(box_num);
        label.filename = filename;
        for (int i = 0; i < box_num; ++i) {
            Box_label_v4 box;
            train_data >> box.x >> box.y >> box.w >> box.h >> box.id;
            box.left = box.x - box.w / 2;
            box.right = box.x + box.w / 2;
            box.top = box.y - box.h / 2;
            box.bottom = box.y + box.h / 2;
            box.track_id = i + (custom_hash(filename.c_str()) % max_obj_img) * max_obj_img;
            label.boxes.push_back(box);
        }
        dataset.push_back(label);
    }
    train_data.close();
}

void YOLOv4_DataLoader::clean_data(const char *in, const char *out) {
    ifstream train_data;
    FILE *f = fopen(out, "w");
    train_data.open(in);
    int count = 1;
    
    while(!train_data.eof()) {
        string filename;
        train_data >> filename;
        int box_num;
        train_data >> box_num;
        yolo_label_v4 label; label.boxes.reserve(box_num);
        label.filename = filename;
        IMG img(filename.c_str());
        if (img.channel == 3 && !img.empty()) {
            printf("Count: %d\n", count++);
            fprintf(f, "%s %d ", filename.c_str(), box_num);
            for (int i = 0; i < box_num; ++i) {
                Box_label_v4 box;
                train_data >> box.x >> box.y >> box.w >> box.h >> box.id;
                fprintf(f, "%f %f %f %f %d ", box.x, box.y, box.w, box.h, box.id);
            }
            fprintf(f, "\n");
        } else {
            for (int i = 0; i < box_num; ++i) {
                Box_label_v4 box;
                train_data >> box.x >> box.y >> box.w >> box.h >> box.id;
            }
        }
    }
    train_data.close();
    fclose(f);
}

void YOLOv4_DataLoader::mark_truth(int index) {
    IMG img(dataset[index].filename.c_str());
    vector<Box_label_v4> &boxes = dataset[index].boxes;
    vector<Detection> det; det.resize(boxes.size());
    for (int i = 0; i < boxes.size(); ++i) {
        Box_label_v4 &box = boxes[i];
        det[i].prob.resize(80);
        det[i].prob[box.id] = 1;
        det[i].objectness = 1;
        det[i].bbox.x = box.x * img.width;
        det[i].bbox.w = box.w * img.width;
        det[i].bbox.y = box.y * img.height;
        det[i].bbox.h = box.h * img.height;
    }
    yolov4_mark(det, img, 80, 0.9);
    img.save();
}

vector<Box_label_v4> YOLOv4_DataLoader::get_label(int index) {
    static auto rng = std::mt19937((unsigned)time(NULL));
    vector<Box_label_v4> &data = dataset[index].boxes;
    shuffle(data.begin(), data.end(), rng);
    
    return data;
}

IMG YOLOv4_DataLoader::get_img(int index) {
    return IMG(dataset[index].filename.c_str());
}

yolo_train_args_v4 YOLOv4_DataLoader::get_train_arg(int index) {
    int net_w = 416;
    int net_h = 416;
    float jitter = 0.3;
    float hue = 0.1;
    float saturation = 1.5;
    float exposure = 1.5;
    
    IMG origin = get_img(index);
    
    int oh = origin.height;
    int ow = origin.width;
    
    int dw = (ow * jitter);
    int dh = (oh * jitter);
    
    float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale;
    float dhue = 0, dsat = 0, dexp = 0;
    
    r1 = Random(0, 1);
    r2 = Random(0, 1);
    r3 = Random(0, 1);
    r4 = Random(0, 1);
    
    r_scale = Random(0, 1);
    
    dhue = Random(-hue, hue);
    dsat = Random_scale(saturation);
    dexp = Random_scale(exposure);
    
    bool flip = (Random() > 0);
    
    int pleft = Random_precal(-dw, dw, r1);
    int pright = Random_precal(-dw, dw, r2);
    int ptop = Random_precal(-dh, dh, r3);
    int pbot = Random_precal(-dh, dh, r4);
    
    float img_ar = (float)ow / (float)oh;
    float net_ar = (float)net_w/ (float)net_h;
    float result_ar = img_ar / net_ar;
    if (result_ar > 1) {
        float oh_tmp = ow / net_ar;
        float delta_h = (oh_tmp - oh) / 2;
        ptop = ptop - delta_h;
        pbot = pbot - delta_h;
    }
    else {
        float ow_tmp = oh * net_ar;
        float delta_w = (ow_tmp - ow) / 2;
        pleft = pleft - delta_w;
        pright = pright - delta_w;
    }
    
    int swidth = ow - pleft - pright;
    int sheight = oh - ptop - pbot;
    
    float sx = (float)swidth / ow;
    float sy = (float)sheight / oh;
    
    IMG cropped = origin.crop(Rect(pleft, ptop, pleft + swidth, ptop + sheight), Scalar(128, 128, 128));
    
    float dx = ((float)pleft / ow) / sx;
    float dy = ((float)ptop / oh) / sy;
    
    IMG resize = cropped.resize(Size(net_w, net_h));
    resize = resize.hsv_distort(dhue, dsat, dexp);
    
    vfloat label_data = get_box(index, dx, dy, 1.0 / sx, 1.0 / sy, flip, net_w, net_h);
    Tensor label(label_data);
//    resize.save("resize.jpg");
    resize = resize.scale(1.0 / 255.0, MAT_32FC3);
    Tensor data(net_w, net_h, 3, 0);
    convert_index_base_to_channel_base((float*)resize.toPixelArray(), data.weight, net_w, net_h, 3);
    
    return yolo_train_args_v4(data, label);
}

vfloat YOLOv4_DataLoader::get_box(int index, float dx, float dy, float sx, float sy, bool flip, int net_w, int net_h) {
    int max_boxes = 90;
    float lowest_w = 1.0 / net_w;
    float lowest_h = 1.0 / net_h;
    vfloat boxes(6 * max_boxes, 0);
    vector<Box_label_v4> box_label = get_label(index);
    correct_box(box_label, dx, dy, sx, sy, flip);
    
    float x, y, w, h;
    int id, track_id;
    int jump = 0;
    for (int i = 0; i < box_label.size(); ++i) {
        x = box_label[i].x;
        y = box_label[i].y;
        w = box_label[i].w;
        h = box_label[i].h;
        id = box_label[i].id;
        track_id = box_label[i].track_id;
        if ((w < lowest_w || h < lowest_h))
            ++jump;
        else {
            boxes[(i - jump) * 6 + 0] = x;
            boxes[(i - jump) * 6 + 1] = y;
            boxes[(i - jump) * 6 + 2] = w;
            boxes[(i - jump) * 6 + 3] = h;
            boxes[(i - jump) * 6 + 4] = id;
            boxes[(i - jump) * 6 + 5] = track_id;
        }
    }
    
    return boxes;
}

void YOLOv4_DataLoader::correct_box(vector<Box_label_v4> &boxes, float dx, float dy, float sx, float sy, bool flip) {
    for (int i = 0; i < boxes.size(); ++i) {
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
            (boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;
        
        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }
        
        boxes[i].left = constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top = constrain(0, 1, boxes[i].top);
        boxes[i].bottom = constrain(0, 1, boxes[i].bottom);
        
        boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
        boxes[i].y = (boxes[i].top + boxes[i].bottom) /2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);
        
        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void YOLOv4_Trainer::train(int epoch) {
    int batch_size = network->getBatchSize();
    auto rng = std::mt19937((unsigned)time(NULL));
    vector<int> index; index.reserve(loader->size());
    float loss = 0;
    size_t data_set_size = loader->size();
    for (int i = 0; i < data_set_size; ++i) {
        index.push_back(i);
    }
        
    yolo_train_args_v4 arg_size = loader->get_train_arg(0);
    int data_size = arg_size.data.size;
    int label_size = arg_size.label.size;
    
    for (int i = 0; i < epoch; ++i) {
        printf("Epoch %d Training[", i + 1);
        loss = 0;
        shuffle(index.begin(), index.end(), rng);
        for (int j = 0; j + batch_size <= data_set_size; ) {
            Tensor data(1, 1, data_size * batch_size, 0);
            float *data_ptr = data.weight;
            Tensor label(1, 1, label_size * batch_size, 0);
            float *label_ptr = label.weight;
                
            for (int k = 0; k < batch_size; ++k, ++j) {
                yolo_train_args_v4 arg = loader->get_train_arg(index[j]);
                float *data_src_ptr = arg.data.weight;
                float *label_src_ptr = arg.label.weight;
                for (int l = 0; l < arg.data.size; ++l) {
                    *(data_ptr++) = *(data_src_ptr++);
                }
                for (int l = 0; l < arg.label.size; ++l) {
                    *(label_ptr++) = *(label_src_ptr++);
                }
            }
            float batch_loss = trainer->train_batch(data, label)[0];
            if (loss == 0) loss = batch_loss;
            loss = loss * 0.9 + batch_loss * 0.1;
            printf("avg_loss: %f batch_loss: %f\n", loss, batch_loss);
            if (j % 5000 >= 0 && j % 5000 < batch_size) {
                network->save(("./backup/backup_" + to_string(j) + ".bin").c_str());
            }
        }
        printf("] ");
        printf("loss: %f\n", loss);
    }
}

//YOLOv4::YOLOv4(int classes_, int batch_size) {
//    classes = classes_;
//    label = get_yolo_label("labelstr.txt", classes);
//    net_width = 416;
//    net_height = 416;
//    threshold = 0.45;
//
//    network = Neural_Network("yolov4");
//    network.addLayer(LayerOption{{"type", "Input"}, {"input_width", "416"}, {"input_height", "416"}, {"input_dimension", "3"}, {"name", "Input"}});
//    // Conv_1
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_1
//
//    // Conv_2
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_2
//
//    // Conv_3
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_3"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_3
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"splits", "2"}, {"split_id", "1"}, {"name", "split_1"}});
//
//    // Conv_4
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_4"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_4
//
//    // Conv_5
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_5"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_5
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_4"}, {"name", "concat_1"}});
//
//    // Conv_6
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_6"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_6
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_6"}, {"name", "concat_2"}, {"input_name", "lr_conv_3"}});
//
//    network.addLayer(LayerOption{{"type", "Pooling"}, {"stride", "2"}, {"kernel_width", "2"}, {"name", "pool_1"}});
//
//    // Conv_7
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_7"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_7
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"splits", "2"}, {"split_id", "1"}, {"name", "split_2"}});
//
//    // Conv_8
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_8"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_8
//
//    // Conv_9
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_9"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_9
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_8"}, {"name", "concat_3"}});
//
//    // Conv_10
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_10"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_10
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_10"}, {"name", "concat_4"}, {"input_name", "lr_conv_7"}});
//
//    network.addLayer(LayerOption{{"type", "Pooling"}, {"stride", "2"}, {"kernel_width", "2"}, {"name", "pool_2"}});
//
//    // Conv_11
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_11"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_11
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"splits", "2"}, {"split_id", "1"}, {"name", "split_3"}});
//
//    // Conv_12
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_12"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_12
//
//    // Conv_13
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_13"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_13
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_12"}, {"name", "concat_5"}});
//
//    // Conv_14
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_14"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_14
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_14"}, {"name", "concat_6"}, {"input_name", "lr_conv_11"}});
//
//    network.addLayer(LayerOption{{"type", "Pooling"}, {"stride", "2"}, {"kernel_width", "2"}, {"name", "pool_3"}});
//
//    // Conv_15
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_15"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_15
//
//    // Conv_16
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_16
//
//    // Conv_lobj_branch
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_lobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_lobj_branch
//    // Conv_lbbox
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_lbbox"}});
//    // End Conv_lbbox
//
//    network.addLayer(LayerOption{{"type", "YOLOv4"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "6"}, {"anchor", "10,14  23,27  37,58  81,82  135,169  344,319"}, {"mask", "3, 4, 5"}, {"max_boxes", "90"}, {"scale_x_y", "1.05"}, {"cls_normalizer", "1.0"}, {"iou_normalizer", "0.07"}, {"iou_loss", "ciou"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"name", "yolo_big"}});
//
//    // Conv_17
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_17"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "lr_conv_16"}});
//    // End Conv_17
//
//    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_1"}});
//
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_14"}, {"name", "concat_7"}});
//
//    // Conv_sobj_branch
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_sobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_sobj_branch
//    // Conv_sbbox
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sbbox"}});
//    // End Conv_sbbox
//    network.addLayer(LayerOption{{"type", "YOLOv4"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "6"}, {"anchor", "10,14  23,27  37,58  81,82  135,169  344,319"}, {"mask", "1, 2, 3"}, {"max_boxes", "90"}, {"scale_x_y", "1.05"}, {"cls_normalizer", "1.0"}, {"iou_normalizer", "0.07"}, {"iou_loss", "ciou"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"name", "yolo_small"}});
//
//    network.addOutput("yolo_small");
//    network.addOutput("yolo_big");
//    network.compile(batch_size);
//    network.load_darknet("yolov4-tiny.weights");
////    network.load_darknet("yolov4-tiny.conv.29");
//    network.shape();
////    network.save("yolov4-tiny.bin");
////    network.save_darknet("yolov4-tiny.conv.29", 64);
//    network.to_prototxt("yolov4-tiny.prototxt");
//}

//YOLOv4::YOLOv4(int classes_, int batch_size) {
//    classes = classes_;
//    label = get_yolo_label("labelstr.txt", classes);
//    net_width = 416;
//    net_height = 416;
//    threshold = 0.45;
//    network = Neural_Network();
//    network.load_otter("yolov4-tiny.otter");
//    network.load_darknet("yolov4-tiny.weights");
//    network.shape();
////    network.save_otter("test.otter");
//}

YOLOv4::YOLOv4(int classes_, int batch_size) {
    classes = classes_;
    label = get_yolo_label("labelstr.txt", classes);
    net_width = 608;
    net_height = 608;
    threshold = 0.45;

    network = Neural_Network("yolov4");
    network.addLayer(LayerOption{{"type", "Input"}, {"input_width", to_string(net_width)}, {"input_height", to_string(net_height)}, {"input_dimension", "3"}, {"name", "Input"}});
    // Conv_1
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_1"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_1

    // Conv_2
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_2"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_2

    // Conv_3
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_3"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_3

    // Conv_4
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_4"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"input_name", "mi_conv_2"}});
    // End Conv_4

    // Conv_5
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_5"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_5

    // Conv_6
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_6"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_6

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_4"}, {"name", "sc_1"}});

    // Conv_7
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_7"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_7

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_3"}, {"name", "concat_1"}});

    // Conv_8
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_8"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_8

    // Conv_9
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_9"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_9

    // Conv_10
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_10"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_10

    // Conv_11
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_11"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"input_name", "mi_conv_9"}});
    // End Conv_11

    // Conv_12
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_12"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_12

    // Conv_13
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_13"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_13

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_11"}, {"name", "sc_2"}});

    // Conv_14
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_14"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_14

    // Conv_15
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_15"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_15

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_2"}, {"name", "sc_3"}});


    // Conv_16
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_16"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_16

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_10"}, {"name", "concat_2"}});

    // Conv_17
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_17"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_17

    // Conv_18
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_18"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_18

    // Conv_19
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_19"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_19

    // Conv_20
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_20"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"input_name", "mi_conv_18"}});
    // End Conv_20

    // Conv_21
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_21"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_21

    // Conv_22
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_22"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_22

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_20"}, {"name", "sc_4"}});

    // Conv_23
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_23"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_23

    // Conv_24
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_24"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_24

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_4"}, {"name", "sc_5"}});

    // Conv_25
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_25"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_25

    // Conv_26
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_26"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_26

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_5"}, {"name", "sc_6"}});

    // Conv_27
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_27"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_27

    // Conv_28
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_28"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_28

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_6"}, {"name", "sc_7"}});

    // Conv_29
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_29"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_29

    // Conv_30
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_30"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_30

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_7"}, {"name", "sc_8"}});

    // Conv_31
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_31"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_31

    // Conv_32
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_32"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_32

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_8"}, {"name", "sc_9"}});

    // Conv_33
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_33"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_33

    // Conv_34
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_34"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_34

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_9"}, {"name", "sc_10"}});

    // Conv_35
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_35"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_35

    // Conv_36
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_36"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_36

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_10"}, {"name", "sc_11"}});

    // Conv_37
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_37"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_37

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_19"}, {"name", "concat_3"}});

    // First Branch
    // Conv_38
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_38"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_38

    // Conv_39
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_39"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_39

    // Conv_40
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_40"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_40

    // Conv_41
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_41"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"input_name", "mi_conv_39"}});
    // End Conv_41

    // Conv_42
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_42"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_42

    // Conv_43
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_43"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_43

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_41"}, {"name", "sc_12"}});

    // Conv_44
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_44"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_44

    // Conv_45
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_45"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_45

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_12"}, {"name", "sc_13"}});

    // Conv_46
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_46"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_46

    // Conv_47
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_47"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_47

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_13"}, {"name", "sc_14"}});

    // Conv_48
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_48"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_48

    // Conv_49
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_49"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_49

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_14"}, {"name", "sc_15"}});

    // Conv_50
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_50"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_50

    // Conv_51
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_51"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_51

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_15"}, {"name", "sc_16"}});

    // Conv_52
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_52"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_52

    // Conv_53
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_53"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_53

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_16"}, {"name", "sc_17"}});

    // Conv_54
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_54"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_54

    // Conv_55
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_55"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_55

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_17"}, {"name", "sc_18"}});

    // Conv_56
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_56"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_56

    // Conv_57
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_57"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_57

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_18"}, {"name", "sc_19"}});

    // Conv_58
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_58"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_58

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_40"}, {"name", "concat_4"}});

    // Branch 2
    // Conv_59
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_59"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_59


    // Conv_60
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_60"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_60

    // Conv_61
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_61"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_61

    // Conv_62
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_62"}, {"batchnorm", "true"}, {"activation", "Mish"}, {"input_name", "mi_conv_60"}});
    // End Conv_62

    // Conv_63
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_63"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_63

    // Conv_64
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_64"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_64

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "mi_conv_62"}, {"name", "sc_20"}});

    // Conv_65
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_65"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_65

    // Conv_66
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_66"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_66

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_20"}, {"name", "sc_21"}});

    // Conv_67
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_67"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_67

    // Conv_68
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_68"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_68

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_21"}, {"name", "sc_22"}});

    // Conv_69
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_69"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_69

    // Conv_70
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_70"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_70

    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_22"}, {"name", "sc_23"}});

    // Conv_71
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_71"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_71

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "mi_conv_61"}, {"name", "concat_5"}});

    // Conv_72
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_72"}, {"batchnorm", "true"}, {"activation", "Mish"}});
    // End Conv_72

    // Conv_73
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_73"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_73

    // Conv_74
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_74"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_74

    // Conv_75
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_75"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_75

    // SPP
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "4"}, {"name", "spp_5"}});

    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "9"}, {"stride", "1"}, {"padding", "8"}, {"name", "spp_9"}, {"input_name", "lr_conv_75"}});

    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "13"}, {"stride", "1"}, {"padding", "12"}, {"name", "spp_13"}, {"input_name", "lr_conv_75"}});

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "spp_9, spp_5, lr_conv_75"}, {"name", "concat_6"}});
    // End SPP

    // Conv_76
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_76"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_76

    // Conv_77
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_77"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_77

    // Conv_78
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_78"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_78

    // Conv_79
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_79"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_79

    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_1"}});

    // Conv_80
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_80"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "mi_conv_59"}});
    // End Conv_80

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "upsample_1"}, {"name", "concat_7"}});

    // Conv_81
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_81"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_81

    // Conv_82
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_82"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_82

    // Conv_83
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_83"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_83

    // Conv_84
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_84"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_84

    // Conv_85
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_85"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_85

    // Conv_86
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_86"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_86

    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_2"}});

    // Conv_87
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_87"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "mi_conv_38"}});
    // End Conv_87

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "upsample_2"}, {"name", "concat_8"}});

    // Conv_88
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_88"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_88

    // Conv_89
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_89"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_89

    // Conv_90
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_90"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_90

    // Conv_91
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_91"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_91

    // Conv_92
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_92"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_92

    // Conv_sobj
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_sobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_sobj

    // Conv_sbox
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_sbox"}});
    // End Conv_sbox

    // YOLO small
    network.addLayer(LayerOption{{"type", "YOLOv4"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "12,16 19,36 40,28 36,75 76,55 72,146 142,110 192,243 459,401"}, {"mask", "0, 1, 2"}, {"max_boxes", "200"}, {"scale_x_y", "1.2"}, {"cls_normalizer", "1.0"}, {"iou_normalizer", "0.07"}, {"iou_loss", "ciou"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"iou_thresh", "0.213"}, {"beta_nms", "0.6"}, {"max_delta", "5"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"name", "yolo_small"}});
    // End YOLO small


    // Conv_93
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_93"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "lr_conv_92"}});
    // End Conv_93

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_85"}, {"name", "concat_9"}});

    // Conv_94
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_94"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_94

    // Conv_95
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_95"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_95

    // Conv_96
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_96

    // Conv_97
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_97"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_97

    // Conv_98
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_98"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_98

    // Conv_mobj
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_mobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_mobj

    // Conv_mbox
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_mbox"}});
    // End Conv_mbox

    // YOLO middle
    network.addLayer(LayerOption{{"type", "YOLOv4"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "12,16 19,36 40,28 36,75 76,55 72,146 142,110 192,243 459,401"}, {"mask", "3, 4, 5"}, {"max_boxes", "200"}, {"scale_x_y", "1.1"}, {"cls_normalizer", "1.0"}, {"iou_normalizer", "0.07"}, {"iou_loss", "ciou"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"iou_thresh", "0.213"}, {"beta_nms", "0.6"}, {"max_delta", "5"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"name", "yolo_middle"}});
    // End YOLO middle


    // Conv_99
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "1"}, {"name", "conv_99"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "lr_conv_98"}});
    // End Conv_99

    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_78"}, {"name", "concat_10"}});

    // Conv_100
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_100"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_100

    // Conv_101
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_101"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_101

    // Conv_102
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_102"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_102

    // Conv_103
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_103"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_103

    // Conv_104
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_104"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_104

    // Conv_bobj
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "1"}, {"name", "conv_bobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_bobj

    // Conv_bbox
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "0"}, {"name", "conv_bbox"}});
    // End Conv_bbox

    // YOLO big
    network.addLayer(LayerOption{{"type", "YOLOv4"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "12,16 19,36 40,28 36,75 76,55 72,146 142,110 192,243 459,401"}, {"mask", "6, 7, 8"}, {"max_boxes", "200"}, {"scale_x_y", "1.05"}, {"cls_normalizer", "1.0"}, {"iou_normalizer", "0.07"}, {"iou_loss", "ciou"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"iou_thresh", "0.213"}, {"beta_nms", "0.6"}, {"max_delta", "5"}, {"ignore_thresh", "0.7"}, {"truth_thresh", "1"}, {"name", "yolo_big"}});
    // End YOLO big

    network.addOutput("yolo_small");
    network.addOutput("yolo_middle");
    network.addOutput("yolo_big");
    network.compile(batch_size);
    network.load_darknet("yolov4.weights");
//    network.load_darknet("darknet53.conv.74");
    network.shape();
//    network.show_detail();
//    network.save("yolov4.bin");
    network.to_prototxt("yolov4.prototxt");
//    network.save_otter("yolov4.otter");
//    network.save_ottermodel("yolov4.ottermodel");
}

unsigned long custom_hash(const char *str) {
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}
