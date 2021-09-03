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

    class YOLOv3 nn("yolov3.bin");
    IMG img("5D4A6413.JPG");
    nn.detect(img);
    img.save("detected.jpg");
//    IMG img2("kite.jpg");
//    nn.detect(img2);
//    img2.save("test2.jpg");
//    IMG img3("person.jpg");
//    nn.detect(img3);
//    img3.save("test3.jpg");
//    IMG img4("horses.jpg");
//    nn.detect(img4);
//    img4.save("test4.jpg");
//    IMG img5("5D4A0550_baseline.jpg");
//    nn.detect(img5);
//    img5.save("test5.jpg");
    

//    Mtcnn mtcnn("1630017229_149_212958.062500.bin", "1630045057_23_52704.917969.bin", "1630045562_6_10078.381836.bin");
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
