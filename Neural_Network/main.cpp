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
#include "Mtcnn.hpp"
#include "YOLOv3.hpp"
#include "YOLOv4.hpp"
#include "Test_All_Layer.hpp"
#include "Machine_Learning.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    // This is a good day to learn.
    
//    test_all_layer(true);
    
//    YOLOv4 nn(80, 32);
//
//    YOLOv4_DataLoader loader("train_clear.txt");
//
//    Trainer trainer(&nn.network, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.00261}, {"warmup", 1000}, {"policy", Trainer::Policy::STEPS}, {"steps", 2}, {"steps_1", 1600000}, {"steps_2", 1800000}, {"scals_1", 0.1}, {"scales_2", 0.1}, {"max_batches", 2000200}, {"sub_division",1}, {"l2_decay", 0.0005}});
//
////    yolo_train_args_v4 test = loader.get_train_arg(0);
////    for (int i = 0; i < 100; ++i)
////        trainer.train_batch(test.data, test.label);
//
//    YOLOv4_Trainer t(&nn.network, &trainer, &loader);
//    t.train(1);

//    IMG img = loader.get_img(0);
//    nn.detect(img);
//    img.save("test1.jpg");

//    nn.network.save("testyolo.bin");
    
//    YOLOv3 test(80);
//    test.network.to_prototxt("test.prototxt");
//
//    IMG img("dog.jpg");
//    test.detect(img);
//    img.save("test_train.jpg");
    
//    YOLOv3 yolo("backup_15008_x11.ottermodel");
//    IMG img("33_Running_Running_33_156.jpg");
//    yolo.detect(img);
//    img.save("detected.jpg", 100);
    
//    YOLOv4 yolo("backup_0_40000_v4_r7.ottermodel");
//    IMG img("umbrella.jpg");
//    yolo.detect(img);
//    img.save("detected.jpg", 100);
    
//    IMG img("target.jpg");
//    Clock c;
//    img = img.gaussian_blur(30);
//    c.stop_and_show();
//    img.save("gaussian.jpg");

//    Mtcnn mtcnn("pnet_default.ottermodel", "rnet_default.ottermodel", "onet_new.ottermodel");
//    mtcnn.min_face_size = 0;
//    IMG img("nthuee.jpg");
//    Clock c;
//    vector<Bbox> result = mtcnn.detect(img);
//    c.stop_and_show();
//    mtcnn.mark(img, result, true);
//    img.save("result.jpg", 80);
//    mtcnn.layout(result);
    
//    Neural_Network nn;
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "1"}, {"input_height", "1"}, {"input_dimension", "2"}, {"name", "data"}});
//    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "4"}, {"activation", "Sigmoid"}});
//    nn.addLayer(LayerOption{{"type", "FullyConnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
//    nn.compile();
//    nn.shape();
////    nn.to_prototxt();
////    nn.save_otter("test.otter");
//
//    Tensor a(1, 1, 2, 0);
//    Tensor b(1, 1, 2, 1);
//    Tensor c(1, 1, 2); c = {0, 1};
//    Tensor d(1, 1, 2); d = {1, 0};
//    vtensor data{a, b, c, d};
//    Tensor a_l(1, 1, 1, 0);
//    Tensor b_l(1, 1, 1, 0);
//    Tensor c_l(1, 1, 1, 1);
//    Tensor d_l(1, 1, 1, 1);
//    vtensor label{a_l, b_l, c_l, d_l};
//
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.1}, {"warmup", 5}});
//    trainer.train_batch(data, label, 100);
//
//    printf("Input (0, 0) -> %.0f(%f)\n", nn.predict(&a)[0], nn.predict(&a)[1]);
//    printf("Input (0, 1) -> %.0f(%f)\n", nn.predict(&c)[0], nn.predict(&c)[1]);
//    printf("Input (1, 0) -> %.0f(%f)\n", nn.predict(&d)[0], nn.predict(&d)[1]);
//    printf("Input (1, 1) -> %.0f(%f)\n", nn.predict(&b)[0], nn.predict(&b)[1]);
    
//    Mat a(3, 4, MAT_32FC1);
//    a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//    cout << a;
//    Mat b = a.transpose();
//    cout << b;
    
//    Neural_Network nn;
//    nn.load_otter("yolov4-p6.otter");
//    nn.shape();
//    nn.to_prototxt("yolov4-p6.prototxt");
    
//    YOLOv4 yolo("backup_25024_v4_x13.ottermodel");
//    IMG img("5D4A1809.JPG");
//    yolo.detect(img);
//    img.save("detected.jpg", 100);
    
//    Mat a(3, 3, MAT_32FC1);
//    cout << a;
    
    return 0;
}
