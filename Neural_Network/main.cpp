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
#include "YOLOv3.hpp"
#include "Test_All_Layer.hpp"

using namespace std;
using namespace std::chrono;

int main(int argc, const char * argv[]) {
    // This is a good day to learn.
    
//    test_all_layer(true);

//    YOLOv3 nn(80, 1);
//
//    YOLOv3_DataLoader loader("train_clear.txt");
//
//    Trainer trainer(&nn.network, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.001}, {"warmup", 100}, {"steps", 2}, {"steps_1", 400000}, {"steps_2", 450000}, {"scales_1", 0.1}, {"scales_2", 0.1}, {"max_batches", 500200}, {"sub_division", 2}});
//
////    YOLOv3_Trainer yolo_trainer(&nn.network, &trainer, &loader);
////    yolo_trainer.train(1);
//
//    yolo_train_args test = loader.get_train_arg(0);
//    for (int i = 0; i < 100; ++i)
//        trainer.train_batch(test.data, test.label);
//
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
    
    YOLOv3 yolo(80);
    IMG img("person.jpg");
    yolo.detect(img);
    img.save("detected.jpg", 100);

//    Mtcnn mtcnn("1630830419_149_217474.312500.bin", "1630899594_119_16779.394531.bin", "1631183863_35_2587.741699.bin");
//    mtcnn.min_face_size = 0;
//    IMG img("target.jpg");
//    Clock c;
//    vector<Bbox> result = mtcnn.detect(img);
//    c.stop_and_show();
//    mtcnn.mark(img, result, true);
//    img.save("result.jpg", 80);
//    mtcnn.layout(result);
    
//    Neural_Network nn;
//    nn.load("1631183863_35_2587.741699.bin");
//    nn.to_prototxt();
    
//    Neural_Network nn;
//    nn.addLayer(LayerOption{{"type", "Input"}, {"input_width", "1"}, {"input_height", "1"}, {"input_dimension", "2"}, {"name", "data"}});
//    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "4"}, {"activation", "PRelu"}});
//    nn.addLayer(LayerOption{{"type", "Fullyconnected"}, {"number_neurons", "2"}, {"activation", "Softmax"}});
//    nn.compile();
//    nn.shape();
//    nn.to_prototxt("test.prototxt");
//
//    Tensor a(1, 1, 2, 0);
//    Tensor b(1, 1, 2, 1);
//    Tensor c(1, 1, 2); c = {0, 1};
//    Tensor d(1, 1, 2); d = {1, 0};
//    vtensor data{a, b, c, d};
//    Tensor a_l(1, 1, 1, 1);
//    Tensor b_l(1, 1, 1, 1);
//    Tensor c_l(1, 1, 1, 0);
//    Tensor d_l(1, 1, 1, 0);
//    vtensor label{a_l, b_l, c_l, d_l};
//
//    Trainer trainer(&nn, TrainerOption{{"method", Trainer::Method::ADAM}, {"learning_rate", 0.1}, {"warmup", 5}});
//    trainer.train(data, label, 100);
    
    return 0;
}
