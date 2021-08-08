//
//  main.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include <iostream>
#include <chrono>

#include "Neural_Network.hpp"
#include "Data_Process.hpp"
#include "Image_Process.hpp"
#include "Mtcnn.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, const char * argv[]) {
//    IMG img((argc >= 2) ? argv[1] : "target.jpg");
//
//    Mtcnn mtcnn("pnet_9446_250k.bin", "rnet_v1.bin", "onet_9998_all.bin");
//    mtcnn.min_face_size = (argc >= 3) ? atoi(argv[2]) : 0;
//    vector<Bbox> bbox_list = mtcnn.detect(img);
//
//    mtcnn.mark(img, bbox_list);
//    img.save("result.jpg", 80);
    
//    JPEG img("5D4A0379.JPG");
//    img.save();
//    img.showPicInfo();
    // This is a good day to learn.
    
//        Neural_Network nn;
//         nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "28"}, {"input_height", "28"}, {"input_dimension", "3"}});
//         nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
//         nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//         nn.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "20"}, {"kernel_width", "5"}, {"stride", "1"}, {"padding", "1"}, {"activation", "PRelu"}});
//         nn.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}});
//         nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "3"}, {"activation", "Softmax"}});
////         nn.addOutput("8");
//         nn.makeLayer();
////         nn.load("model.bin");
//         nn.shape();
//
//
//         Data bee("bee.npy", 28, 28);
//         vtensor data_bee = bee.get(500);
//         vector<vfloat> label_bee(500, vfloat(1, 0));
//         Data cat("cat.npy", 28, 28);
//         vtensor data_cat = cat.get(500);
//         vector<vfloat> label_cat(500, vfloat(1, 1));
//         Data fish("fish.npy", 28, 28);
//         vtensor data_fish = fish.get(500);
//         vector<vfloat> label_fish(500, vfloat(1, 2));
//
//         vtensor data_train;
//         vector<vfloat> data_label;
//         for (int i = 0; i < 300; ++i) {
//             data_train.push_back(data_bee[i]);
//             data_train.push_back(data_cat[i]);
//             data_train.push_back(data_fish[i]);
//             data_label.push_back(label_bee[i]);
//             data_label.push_back(label_cat[i]);
//             data_label.push_back(label_fish[i]);
//         }
//
//         vtensor data_valid;
//         vector<vfloat> label_valid;
//         for (int i = 300; i < 500; ++i) {
//             data_valid.push_back(data_bee[i]);
//             data_valid.push_back(data_cat[i]);
//             data_valid.push_back(data_fish[i]);
//             label_valid.push_back(label_bee[i]);
//             label_valid.push_back(label_cat[i]);
//             label_valid.push_back(label_fish[i]);
//         }
//
//         printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label) * 100);
//
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"batch_size", 2}, {"learning_rate", 0.001}});
//
//    auto start = high_resolution_clock::now();
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<milliseconds>(stop - start);
//         trainer.train(data_train, data_label, 1);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//         //nn.train("SVG", 0.001, data_train, data_label, 1);
//
//         printf("Accuracy: %.2f%%\n", nn.evaluate(data_train, data_label) * 100);
////         nn.shape();
//
//         vfloat out = nn.predict(&data_bee[0]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_bee[327]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_bee[376]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//         out = nn.predict(&data_cat[15]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_cat[312]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_cat[305]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//
//         out = nn.predict(&data_fish[98]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_fish[312]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
//         out = nn.predict(&data_fish[456]);
//         printf("Predict: %.0f (%.2f%%)\n", out[0], out[1] * 100);
    
    
    
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
    
    //    RNet rnet("rnet_ensure_9721.bin");
    //
    //    string dir = "/Users/chenjunhao/Desktop/mtcnn/";
    //    string data = "DATA/WIDER_train/images/";
    //    string annotate = "MTCNN-Tensorflow-master/prepare_data/bbox_list.txt";
    //    string data_list_path = dir + annotate;
    //
    //    string out_path = dir + "MTCNN-Tensorflow-master/prepare_data/bbox_list_rnet.txt";
    //    FILE *out = fopen(out_path.c_str(), "w");
    //    fstream f;
    //    f.open(data_list_path.c_str());
    //    int count = 0;
    //    while(!f.eof()) {
    //        string filename;
    //        f >> filename;
    //        string data_path = dir + data + filename;
    //        cout << data_path << endl;
    //        char a, b;
    //        f.get(a);
    //        f.get(b);
    //        vector<Bbox> bbox_list;
    //        while (b != '\n') {
    //            f.putback(b);
    //            f.putback(a);
    //            int c, d, e, g;
    //            float h;
    //            f >> c >> d >> e >> g >> h;
    ////            printf("%d %d %d %d %f\n", c, d, e, g, h);
    //            Bbox box(c, d, e, g, h, 0, 0, 0, 0);
    //            bbox_list.push_back(box);
    //            f.get(a);
    //            f.get(b);
    //        }
    //        count++;
    //        printf("list length: %d\n", (int)bbox_list.size());
    //        IMG img(data_path.c_str());
    //        vector<Bbox> bbox = rnet.detect(img, bbox_list);
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
    
    
    
    
//    PNet pnet("pnet_9446_250k.bin");
//    pnet.min_face_size = 50;
//    pnet.threshold[0] = 0.97;
//    IMG img("pic1.jpg");
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
//        pnet_detect.drawRectangle(Rect{(bbox[i].x1), (bbox[i].y1), (bbox[i].x2), (bbox[i].y2)}, RED);
//    }
//    pnet_detect.save("pnet_predict.jpg", 80);
//
//
//    RNet rnet("rnet_v1.bin");
//    rnet.threshold[0] = 0.7;
//
//    start = high_resolution_clock::now();
//    vector<Bbox> rnet_bbox = rnet.detect(img, bbox);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("RNet Get %d proposal box! time: %lldms\n", (int)rnet_bbox.size(), duration.count());
//
//    IMG rnet_detect(img);
//    for (int i = 0; i < rnet_bbox.size(); ++i) {
//        rnet_detect.drawRectangle(Rect{(rnet_bbox[i].x1), (rnet_bbox[i].y1), (rnet_bbox[i].x2), (rnet_bbox[i].y2)}, RED);
//    }
//    rnet_detect.save("rnet_predict.jpg", 80);
//
//    ONet onet("onet_9998_all.bin");
//    onet.threshold[0] = 0.8;
//
//    start = high_resolution_clock::now();
//    vector<Bbox> onet_bbox = onet.detect(img, rnet_bbox);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("ONet Get %d proposal box! time: %lldms\n", (int)onet_bbox.size(), duration.count());
//
//    IMG onet_detect(img);
//    for (int i = 0; i < onet_bbox.size(); ++i) {
//        int radius = min(onet_bbox[i].x2 - onet_bbox[i].x1 + 1, onet_bbox[i].y2 - onet_bbox[i].y1 + 1) / 30 + 1;
//        onet_detect.drawRectangle(Rect{(onet_bbox[i].x1), (onet_bbox[i].y1), (onet_bbox[i].x2), (onet_bbox[i].y2)}, RED, radius);
//        onet_detect.drawCircle(Point(onet_bbox[i].lefteye_x, onet_bbox[i].lefteye_y), RED, radius);
//        onet_detect.drawCircle(Point(onet_bbox[i].righteye_x, onet_bbox[i].righteye_y), RED, radius);
//        onet_detect.drawCircle(Point(onet_bbox[i].nose_x, onet_bbox[i].nose_y), RED, radius);
//        onet_detect.drawCircle(Point(onet_bbox[i].leftmouth_x, onet_bbox[i].leftmouth_y), RED, radius);
//        onet_detect.drawCircle(Point(onet_bbox[i].rightmouth_x, onet_bbox[i].rightmouth_y), RED, radius);
//    }
//    onet_detect.save("onet_predict.jpg", 80);
    
//    Mtcnn mtcnn("pnet_9489_all.bin", "rnet_9885.bin", "onet_9998_all.bin");
//    mtcnn.min_face_size = 50;
//    IMG img("pic1.jpg");
//    vector<Bbox> result = mtcnn.detect(img);
//    mtcnn.mark(img, result);
//    img.save("result.jpg", 80);
//    mtcnn.layout(result);
    
//    IMG img("pic1.jpg");
//    auto start = high_resolution_clock::now();
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<milliseconds>(stop - start);
//    img = img.gaussian_blur(3);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//    img.save("gaussian.jpg");
//    Kernel k(3, 3, 1, 1);
//    k = {0, 1, 0, 1, -4, 1, 0, 1, 0};
//    Kernel g(1, 1, 1, 3);
//    IMG gray = img.convertGray();
//    IMG lap = gray.filter(1, k, false);
//    lap = lap.filter(1, g, false);
//    lap.save("lap.jpg");
//    IMG sobel = gray.sobel();
//    sobel.save("sobel.jpg");
//    lap.histogram(Size(1000, 500), 1, "lap_histo.jpg");
//    sobel.histogram(Size(1000, 500), 1, "sobel_histo.jpg");
    
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
    
//    vtensor data_set_pnet;
//    vector<vfloat> label_set_pnet;
//    mtcnn_data_loader("img_data_250k.bin", "label_data_250k.bin", data_set_pnet, label_set_pnet, 12, 1429442);
//    PNet pnet;
//    Trainer trainer(&pnet.pnet, TrainerOption{{"method", Trainer::Method::ADAM}, {"eps", 1e-14}, {"batch_size", 384}, {"learning_rate", 0.001}});
//    trainer.train(data_set_pnet, label_set_pnet, 240);
//
//    mtcnn_evaluate(&pnet.pnet, data_set_pnet, label_set_pnet);
    
    
    //    IMG img("Carlos_Barra_0001.jpg");
    
//    Mat mat(3, 3, MAT_32FC3), mat2(3, 3, MAT_32FC3);
//
//    unsigned char *ptr = mat.ptr();
//    int step = mat.getStep();
//    int pix_size = mat.elemSize();
//    unsigned char *act_ptr;
//    Scalar test(3, 4, 5);
//    double buf[4];
//    convertScalar(test, buf, MAT_32FC3);
//
//    printf("pix_size: %d\n", pix_size);
//
//    for (int i = 0; i < mat.height; ++i) {
//        for (int j = 0; j < mat.width; ++j) {
//            act_ptr = ptr + i * step + j * pix_size;
//            for (int k = 0; k < pix_size; ++k) {
//                act_ptr[k] = ((unsigned char *)buf)[k];
//            }
//        }
//    }
//
////    Mat::MatIterator<Vec3f> it = mat.begin<Vec3f>();
////    for ( ; it != mat.end<Vec3f>(); ++it) {
////        printf("(%.2f %.2f %.2f) ", (*it)[0], (*it)[1], (*it)[2]);
////    }
//
//    for (int i = 0; i < mat.height; ++i) {
//        for (int j = 0; j < mat.width; ++j) {
//            printf("(%.2f %.2f %.2f) ", mat.at<Vec3f>(i, j)[0], mat.at<Vec3f>(i, j)[1], mat.at<Vec3f>(i, j)[2]);
//        }
//        printf("\n");
//    }
//
//
//    IMG img("pic1.jpg");
////    img = img.convertGray();
//    Mat kernel(3, 3, MAT_32FC1, Scalar(1.0 / 9));
//    auto start = high_resolution_clock::now();
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<milliseconds>(stop - start);
//    img = img.filter(kernel, MAT_8UC3);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//    img.convertTo(MAT_8UC3);
//    img.save("test.jpg");
    
//    Mat a(3, 3, MAT_8UC1, Scalar(3));
//    Mat b(3, 3, MAT_8UC1, Scalar(5));
//    Mat c = a.subtract(b, MAT_8UC1);
//
//    for (auto it = c.begin<unsigned char>(); it != c.end<unsigned char>(); ++it) {
//        printf("%d ", *it);
//    }
    
    IMG img("pic1.jpg");
    img.save("test.jpg");
//
//    start = high_resolution_clock::now();
//    IMG crop = img.crop(Rect(2476, 1431, 2547, 1519));
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//    crop.save("crop.jpg");
//
//    start = high_resolution_clock::now();
//    IMG resize = crop.resize(Size(0, 0), 0.5, 0.5);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//    resize.save("resize.jpg");
//
//    Kernel k(5, 5, 1, 1);
//    start = high_resolution_clock::now();
//    img = img.filter(k);
//    stop = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(stop - start);
//    printf("Time: %lldms\n", duration.count());
//    img.save("filter.jpg");
    
//    Mat a(3, 3, MAT_32SC3);
//
//    int count = 0;
//    for (auto it = a.begin<Vec3i>(); it != a.end<Vec3i>(); ++it) {
//        (*it)[0] = count++; (*it)[1] = count++; (*it)[2] = count++;
//        count += 50;
//        printf("(%d %d %d)", (*it)[0], (*it)[1], (*it)[2]);
//    }
//    printf("\n");
//
//    Mat b = a.convertTo(MAT_32UC3, 0, 0);
//
//    for (auto it = b.begin<Vec3u>(); it != b.end<Vec3u>(); ++it) {
//        printf("(%u %u %u)", (*it)[0], (*it)[1], (*it)[2]);
//    }
    
//    IMG img("pic1.jpg");
//    img.histogram(Size(1000, 500), 1, "rgb_histo.jpg");
//    img = img.convertGray();
//    img.histogram(Size(1000, 500), 1, "gray_histo.jpg");
//    Mat kernel(5, 5, MAT_32FC1, Scalar(1.0 / 25));
//    img = img.convertGray();
//    img = img.gaussian_blur(50);
//    img.save("filter.jpg");
//    IMG sobel = img.sobel();
//    IMG thres = sobel.threshold(127, 255);
//    Kernel d(3, 3, 1, 1);
//    IMG dilate = thres.dilate(d);
//    IMG erode = thres.erode(d);
//    IMG open = thres.opening(d);
//    IMG close = thres.closing(d);
//    sobel.save("sobel.jpg");
//    thres.save("thres.jpg");
//    dilate.save("dilate.jpg");
//    erode.save("erode.jpg");
//    open.save("open.jpg");
//    close.save("close.jpg");
    
    return 0;
}
