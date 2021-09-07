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
    
//    test_all_layer();

//    class YOLOv3 nn(80);
//    YOLOv3_DataLoader loader("train.txt");
//
//    Trainer trainer(&nn.network, TrainerOption{{"method", Trainer::Method::SGD}, {"learning_rate", 0.0001}});
//
//    for (int i = 0; i < 3; ++i) {
//        yolo_train_args arg = loader.get_train_arg(i);
//        float loss = trainer.train(arg.data, arg.label)[0];
//        printf("loss: %f\n", loss);
//    }

    
    class YOLOv3 yolo("yolov3.bin");
    IMG img("IMG_2204.jpg");
    yolo.detect(img);
    img.save("detected.jpg", 100);

//    Mtcnn mtcnn("1630961845_149_222280.687500.bin", "1630899594_119_16779.394531.bin", "1630914915_41_3235.218994.bin");
//    mtcnn.min_face_size = 0;
//    IMG img("target.jpg");
//    Clock c;
//    vector<Bbox> result = mtcnn.detect(img);
//    c.stop_and_show();
//    mtcnn.mark(img, result, true);
//    img.save("result.jpg", 80);
//    mtcnn.layout(result);
    return 0;
}
