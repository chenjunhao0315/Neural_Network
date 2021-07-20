//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>
#include <chrono>

#include "Neural_Network.hpp"
//#include "Data_Process.hpp"
#include "Image_Process.hpp"
#include "Mtcnn.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, const char * argv[]) {
    // This is a good day to learn.
    
//    vtensor data_set_pnet;
//    vector<vfloat> label_set_pnet;
//
//    mtcnn_data_loader("img_data_pnet.bin", "label_data_pnet.bin", data_set_pnet, label_set_pnet, 12, 100000);
//
//
//    PNet pnet;
//    mtcnn_evaluate(&pnet.pnet, data_set_pnet, label_set_pnet);
//
//    Trainer trainer_pnet(&pnet.pnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"eps", 1e-14}, {"batch_size", 384}, {"learning_rate", 0.0005}});
//    trainer_pnet.train(data_set_pnet, label_set_pnet, 15);
//
//    mtcnn_evaluate(&pnet.pnet, data_set_pnet, label_set_pnet);
//
//    vfloat out = pnet.pnet.Forward(&data_set_pnet[0]);
//    printf("Predict cls: %.2f %.2f\nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
//    out = label_set_pnet[0];
//    printf("Label cls: %.2f \nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14]);
    
//    PNet pnet("pnet_9446_250k.bin");
//
//    string dir = "/Users/chenjunhao/Desktop/mtcnn/";
//    string data = "DATA/WIDER_train/images/";
//    string annotate = "MTCNN-Tensorflow-master/prepare_data/wider_face_train_bbx_gt.txt";
//    string data_list_path = dir + annotate;
//
//    string out_path = dir + "MTCNN-Tensorflow-master/prepare_data/bbox_list.txt";
//    FILE *out = fopen(out_path.c_str(), "w");
//    fstream f;
//    f.open(data_list_path.c_str());
//    int count = 0;
//    while(!f.eof()) {
//        string filename;
//        f >> filename;
//        string data_path = dir + data + filename;
//        cout << data_path << endl;
//        int n;
//        f >> n;
//        int discard;
//        for (int i = 0; i < n; ++i) {
//            for (int j = 0; j < 10; ++j) {
//                f >> discard;
//            }
//        }
//        count++;
//        IMG img(data_path.c_str());
//        vector<Bbox> bbox = pnet.detect(img);
//        fprintf(out, "%s ", filename.c_str());
//        for (int i = 0; i < bbox.size(); ++i) {
//            fprintf(out, "%d %d %d %d %f ", bbox[i].x1, bbox[i].y1, bbox[i].x2, bbox[i].y2, bbox[i].score);
//        }
//        fprintf(out, "\n");
//        printf("pic: %d / 12880\n", count);
//    }
//    cout << "total: " << count << endl;
//    fclose(out);
//    f.close();
//
//    PNet pnet("pnet_9446_250k.bin");
//    pnet.min_face_size = 25;
//    pnet.threshold[0] = 0.93;
//    IMG img("Carlos_Barra_0001.jpg");
//
//    auto start = high_resolution_clock::now();
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<milliseconds>(stop - start);
//    vector<Bbox> bbox = pnet.detect(img);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("PNet Get %d proposal box! time: %lldms\n", (int)bbox.size(), duration.count());
//
//    IMG pnet_detect(img);
//    for (int i = 0; i < bbox.size(); ++i) {
//        pnet_detect.drawRectangle(Rect{(bbox[i].x1), (bbox[i].y1), (bbox[i].x2), (bbox[i].y2)}, Color(255, 0, 0));
//    }
//    pnet_detect.save("pnet_predict.jpg", 80);
//
//
//    RNet rnet("rnet_9747_all.bin");
//    rnet.threshold[0] = 0.6;
//
//    start = high_resolution_clock::now();
//    vector<Bbox> rnet_bbox = rnet.detect(img, bbox);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("RNet Get %d proposal box! time: %lldms\n", (int)rnet_bbox.size(), duration.count());
//
//    IMG rnet_detect(img);
//    for (int i = 0; i < rnet_bbox.size(); ++i) {
//        rnet_detect.drawRectangle(Rect{(rnet_bbox[i].x1), (rnet_bbox[i].y1), (rnet_bbox[i].x2), (rnet_bbox[i].y2)}, Color(255, 0, 0));
//    }
//    rnet_detect.save("rnet_predict.jpg", 80);
    
    
//    vtensor data_set_rnet;
//    vector<vfloat> label_set_rnet;
//    mtcnn_data_loader("img_data_rnet.bin", "label_data_rnet.bin", data_set_rnet, label_set_rnet, 24, 2279);
//
////    vfloat out = label_set_rnet[2278];
////    printf("Label cls: %.2f \nbbox: %.2f %.2f %.2f %.2f\nlandmark: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14]);
//
//    RNet rnet;
//
//    Trainer trainer(&rnet.rnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"eps", 1e-14}, {"batch_size", 4}, {"learning_rate", 0.001}});
//    trainer.train(data_set_rnet, label_set_rnet, 2);
//
//    mtcnn_evaluate(&rnet.rnet, data_set_rnet, label_set_rnet);
    
    
//    IMG img("Carlos_Barra_0001.jpg");
    
    return 0;
}
