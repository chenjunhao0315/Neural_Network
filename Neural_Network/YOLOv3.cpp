//
//  YOLOv3.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/24.
//

#include "YOLOv3.hpp"

//YOLOv3::YOLOv3(int classes_) {
//    classes = classes_;
//    network = Neural_Network();
//    net_width = 416;
//    net_height = 416;
//    
//    // Revise DarkNet53
//    // Input
//    network.addLayer(LayerOption{{"type", "Input"}, {"input_width", "416"}, {"input_height", "416"}, {"input_dimension", "3"}, {"name", "Input"}});
//    // Conv_1 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End of Conv_1
//    // Downsample Conv_2 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End of Conv_2
//    // Residual Block 1
//    // Conv_3(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_3"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_4(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_4"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_2"}, {"name", "sc_1"}});
//    // End Residual Block 1
//    // Downsample Conv_5 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_5"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_5
//    // Residual Block 2
//    // Conv_6(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_6"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_7(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_7"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_5"}, {"name", "sc_2"}});
//    // End Residual Block 2
//    // Residual Block 3
//    // Conv_8(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_8"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_9(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_9"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_2"}, {"name", "sc_3"}});
//    // End Residual Block 3
//    // Downsample Conv_10 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_10"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_10
//    // Residual Block 4
//    // Conv_11(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_11"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_12(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_12"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_10"}, {"name", "sc_4"}});
//    // End Residual Block 4
//    // Residual Block 5
//    // Conv_13(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_13"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_14(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_14"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_4"}, {"name", "sc_5"}});
//    // End Residual Block 5
//    // Residual Block 6
//    // Conv_15(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_15"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_16(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_5"}, {"name", "sc_6"}});
//    // End Residual Block 6
//    // Residual Block 7
//    // Conv_17(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_17"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_18(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_18"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_6"}, {"name", "sc_7"}});
//    // End Residual Block 7
//    // Residual Block 8
//    // Conv_19(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_19"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_20(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_20"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_7"}, {"name", "sc_8"}});
//    // End Residual Block 8
//    // Residual Block 9
//    // Conv_21(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_21"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_22(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_22"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_8"}, {"name", "sc_9"}});
//    // End Residual Block 9
//    // Residual Block 10
//    // Conv_23(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_23"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_24(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_24"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_9"}, {"name", "sc_10"}});
//    // End Residual Block 10
//    // Residual Block 11
//    // Conv_25(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_25"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_26(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_26"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_10"}, {"name", "route_1"}});
//    // End Residual Block 11
//    // Extract 1
//    // Downsample Conv_27 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_27"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_27
//    // Residual Block 12
//    // Conv_28(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_28"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_29(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_29"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "route_1"}, {"name", "sc_12"}});
//    // End Residual Block 12
//    // Residual Block 13
//    // Conv_30(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_30"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_31(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_31"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_12"}, {"name", "sc_13"}});
//    // End Residual Block 13
//    // Residual Block 14
//    // Conv_32(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_32"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_33(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_33"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_13"}, {"name", "sc_14"}});
//    // End Residual Block 14
//    // Residual Block 15
//    // Conv_34(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_34"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_35(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_35"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_14"}, {"name", "sc_15"}});
//    // End Residual Block 15
//    // Residual Block 16
//    // Conv_36(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_36"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_37(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_37"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_15"}, {"name", "sc_16"}});
//    // End Residual Block 16
//    // Residual Block 17
//    // Conv_38(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_38"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_39(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_39"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_16"}, {"name", "sc_17"}});
//    // End Residual Block 17
//    // Residual Block 18
//    // Conv_40(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_40"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_41(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_41"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_17"}, {"name", "sc_18"}});
//    // End Residual Block 18
//    // Residual Block 19
//    // Conv_42(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_42"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_43(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_43"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_18"}, {"name", "route_2"}});
//    // End Residual Block 19
//    // Extract 2
//    // Downsample Conv_44 with BatchNorm, LRelu
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_44"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_44
//    // Residual Block 20
//    // Conv_45(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_45"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_46(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_46"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "route_2"}, {"name", "sc_20"}});
//    // End Residual Block 20
//    // Residual Block 21
//    // Conv_47(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_47"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_48(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_48"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_20"}, {"name", "sc_21"}});
//    // End Residual Block 21
//    // Residual Block 22
//    // Conv_49(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_49"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_50(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_50"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_21"}, {"name", "sc_22"}});
//    // End Residual Block 22
//    // Residual Block 23
//    // Conv_51(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_51"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_52(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_52"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_22"}, {"name", "sc_23"}});
//    // End Residual Block 23
//    // End Revise DarkNet53
//    
//    // Conv_53(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_53"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_54(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_54"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_55(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_55"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_56(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_56"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Conv_57(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_57"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // Extract 3
//    // Conv_lobj_branch(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_lobj_branch
//    // Conv_lbbox(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lbbox"}});
//    network.addLayer(LayerOption{{"type", "YOLOv3"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "10,13  16,30  33,23  30,61  62,45  59,119  116,90  156,198  373,326"}, {"mask", "6, 7, 8"}, {"name", "yolo_large"}});
//    // End Conv_lbbox
//    
//    // Conv_58(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_58"}, {"input_name", "lr_conv_57"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_58
//    // UpSample
//    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_1"}});
//    // End UpSample
//    // Concat
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "route_2"}, {"name", "concat_1"}});
//    // End Concat
//    // Conv_59(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_59"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_59
//    // Conv_60(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_60"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_60
//    // Conv_61(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_61"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_61
//    // Conv_62(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_62"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_62
//    // Conv_63(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_63"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_63
//    // Extract 4
//    // Conv_mobj_branch(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_mobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_mobj_branch
//    // Conv_mbbox(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_mbbox"}});
//    network.addLayer(LayerOption{{"type", "YOLOv3"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "10,13  16,30  33,23  30,61  62,45  59,119  116,90  156,198  373,326"}, {"mask", "3, 4, 5"}, {"name", "yolo_middle"}});
//    // End Conv_mbbox
//    
//    // Conv_64(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_64"}, {"input_name", "lr_conv_61"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_64
//    // UpSample
//    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_2"}});
//    // End UpSample
//    // Concat
//    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "route_1"}, {"name", "concat_2"}});
//    // End Concat
//    
//    // Conv_65(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_65"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_65
//    // Conv_66(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_66"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_66
//    // Conv_67(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_67"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_67
//    // Conv_68(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_68"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_68
//    // Conv_69(1 * 1)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_69"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_69
//    // Conv_sobj_branch(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    // End Conv_sobj_branch
//    // Conv_sbbox(3 * 3)
//    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sbbox"}});
//    network.addLayer(LayerOption{{"type", "YOLOv3"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "9"}, {"anchor", "10,13  16,30  33,23  30,61  62,45  59,119  116,90  156,198  373,326"}, {"mask", "0, 1, 2"}, {"name", "yolo_small"}});
//    // End Conv_sbbox
//    
//    network.addOutput("yolo_large");
//    network.addOutput("yolo_middle");
//    network.addOutput("yolo_small");
//    network.compile(1);
//    network.shape();
////    network.save("test.bin");
//    threshold = 0.5;
//}

YOLOv3::YOLOv3(int classes_) {
    classes = classes_;
    net_width = 416;
    net_height = 416;
    
    network = Neural_Network("yolov3");
    network.addLayer(LayerOption{{"type", "Input"}, {"input_width", "416"}, {"input_height", "416"}, {"input_dimension", "3"}, {"name", "Input"}});
    // Conv_1
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "16"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_1
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_1"}});
    // Conv_2
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_2
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_2"}});
    // Conv_3
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_3"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_3
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_3"}});
    // Conv_4
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_4"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_4
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_4"}});
    // Conv_5
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_5"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_5
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "2"}, {"name", "pool_5"}});
    // Conv_6
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_6"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_6
    network.addLayer(LayerOption{{"type", "Pooling"}, {"kernel_width", "2"}, {"stride", "1"}, {"padding", "same"}, {"name", "pool_6"}});
    // Conv_7
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_7"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_7
    // Conv_8
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_8"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_8
    
    // Conv_lobj_branch
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_lobj_branch
    // Conv_lbbox
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lbbox"}});
    // End Conv_lbbox
    network.addLayer(LayerOption{{"type", "YOLOv3"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "6"}, {"anchor", "10,14  23,27  37,58  81,82  135,169  344,319"}, {"mask", "3, 4, 5"}, {"max_boxes", "90"}, {"name", "yolo_small"}});
    
    // Conv_9
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_9"}, {"batchnorm", "true"}, {"activation", "LRelu"}, {"input_name", "lr_conv_8"}});
    // End Conv_9
    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_1"}});
    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "lr_conv_5"}, {"name", "concat_1"}});

    // Conv_sobj_branch
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sobj"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    // End Conv_sobj_branch
    // Conv_sbbox
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (classes + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sbbox"}});
    // End Conv_sbbox
    network.addLayer(LayerOption{{"type", "YOLOv3"}, {"classes", to_string(classes)}, {"anchor_num", "3"}, {"total_anchor_num", "6"}, {"anchor", "10,14  23,27  37,58  81,82  135,169  344,319"}, {"mask", "0, 1, 2"}, {"max_boxes", "90"}, {"name", "yolo_big"}});

    network.addOutput("yolo_small");
    network.addOutput("yolo_big");
    network.compile();
    network.load_darknet("yolov3-tiny.weights");
    network.shape();
}

YOLOv3::YOLOv3(const char *model_name) {
    network = Neural_Network();
    network.load(model_name);
    threshold = 0.5;
}

vector<Detection> YOLOv3::detect(IMG &input) {
    IMG src_img = yolo_pre_process_img(input, 416, 416);
    Tensor src_tensor(net_width, net_height, 3, 0);
    convert_index_base_to_channel_base((float *)src_img.toPixelArray(), src_tensor.weight, 416, 416, 3);
    
    Clock c;
    vtensorptr feature_map = network.Forward(&src_tensor);
    c.stop_and_show();
    
    vector<Detection> dets;
    for (int i = 0; i < feature_map.size(); ++i) {
        vector<Detection> det = yolo_correct_box(feature_map[i], input.width, input.height, net_width, net_height, 0);
        dets.insert(dets.end(), det.begin(), det.end());
    }
    
    yolo_nms(dets, classes, threshold);
    yolo_mark(dets, input, 80, 0.5);
    input.save("?.jpg");
    
    
    return vector<Detection>();
}

void yolo_mark(vector<Detection> &dets, IMG &img, int classes, float threshold) {
    for (int i = 0; i < dets.size(); ++i) {
        int sort_class = -1;
        for (int c = 0; c < classes; ++c) {
            if (dets[i].prob[c] > threshold)
                sort_class = 1;
        }
        if (sort_class > 0 && dets[i].objectness > 0)
            img.drawRectangle(Rect(dets[i].bbox.x - dets[i].bbox.w / 2, dets[i].bbox.y - dets[i].bbox.h / 2, dets[i].bbox.x + dets[i].bbox.w / 2, dets[i].bbox.y + dets[i].bbox.h / 2), RED);
    }
}

void YOLOv3::yolo_nms(vector<Detection> &det_list, int classes ,float threshold) {
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

vector<Detection> YOLOv3::yolo_correct_box(Tensor *box_list, int img_w, int img_h, int net_w, int net_h, bool relative) {
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

IMG YOLOv3::yolo_pre_process_img(IMG &img, int net_w, int net_h) {
    IMG canvas(net_w, net_h, MAT_8UC3, Scalar(127, 127, 127));
    
    int img_w  = img.width, img_h = img.height;
    int new_w = int(img_w * min((float)net_w / img_w, (float)net_h / img_h));
    int new_h = int(img_h * min((float)net_w / img_w, (float)net_h / img_h));
    
    IMG resize = img.resize(Size(new_w, new_h));
    canvas.paste(resize, Point((net_w - new_w) / 2, (net_h - new_h) / 2));
//    Mat gain(1, 1, MAT_32FC1, Scalar(0.00392157));
//    canvas = canvas.filter(gain, MAT_32FC3);
    canvas = canvas.scale(0.00392157, MAT_32FC3);
    return canvas;
}

YOLOv3_DataLoader::~YOLOv3_DataLoader() {
    dataset.clear();
}

YOLOv3_DataLoader::YOLOv3_DataLoader(const char *filename) {
    ifstream train_data;
    train_data.open("train.txt");
    
    while(!train_data.eof()) {
        string filename;
        train_data >> filename;
        int box_num;
        train_data >> box_num;
        yolo_label label; label.boxes.reserve(box_num);
        label.filename = filename;
        for (int i = 0; i < box_num; ++i) {
            Box_label box;
            train_data >> box.x >> box.y >> box.w >> box.h >> box.id;
            box.left = box.x - box.w / 2;
            box.right = box.x + box.w / 2;
            box.top = box.y - box.h / 2;
            box.bottom = box.y + box.h / 2;
            label.boxes.push_back(box);
        }
        dataset.push_back(label);
    }
    train_data.close();
}

void YOLOv3_DataLoader::mark_truth(int index) {
    IMG img(dataset[index].filename.c_str());
    vector<Box_label> &boxes = dataset[index].boxes;
    vector<Detection> det; det.resize(boxes.size());
    for (int i = 0; i < boxes.size(); ++i) {
        Box_label &box = boxes[i];
        det[i].prob.resize(80);
        det[i].prob[box.id] = 1;
        det[i].objectness = 1;
        det[i].bbox.x = box.x * img.width;
        det[i].bbox.w = box.w * img.width;
        det[i].bbox.y = box.y * img.height;
        det[i].bbox.h = box.h * img.height;
    }
    yolo_mark(det, img, 80, 0.9);
    img.save();
}

vector<Box_label> YOLOv3_DataLoader::get_label(int index) {
    static auto rng = std::mt19937((unsigned)time(NULL));
    vector<Box_label> &data = dataset[index].boxes;
    shuffle(data.begin(), data.end(), rng);
    
    return data;
}

IMG YOLOv3_DataLoader::get_img(int index) {
    return IMG(dataset[index].filename.c_str());
}

yolo_train_args YOLOv3_DataLoader::get_train_arg(int index) {
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
    
    IMG cropped = origin.crop(Rect(pleft, ptop, pleft + swidth, ptop + sheight));
    
    float dx = ((float)pleft / ow) / sx;
    float dy = ((float)ptop / oh) / sy;
    
    IMG resize = cropped.resize(Size(net_w, net_h));
    resize = resize.hsv_distort(dhue, dsat, dexp);
    
    vfloat label = get_box(index, dx, dy, 1.0 / sx, 1.0 / sy, flip, net_w, net_h);
    
    resize = resize.scale(1.0 / 255.0, MAT_32FC3);
    Tensor data(net_w, net_h, 3, 0);
    convert_index_base_to_channel_base((float*)resize.toPixelArray(), data.weight, net_w, net_h, 3);
    
    return yolo_train_args(data, label);
}

vfloat YOLOv3_DataLoader::get_box(int index, float dx, float dy, float sx, float sy, bool flip, int net_w, int net_h) {
    int max_boxes = 90;
    float lowest_w = 1.0 / net_w;
    float lowest_h = 1.0 / net_h;
    vfloat boxes(5 * max_boxes, 0);
    vector<Box_label> box_label = get_label(index);
    correct_box(box_label, dx, dy, sx, sy, flip);
    
    float x, y, w, h;
    int id;
    int jump = 0;
    for (int i = 0; i < box_label.size(); ++i) {
        x = box_label[i].x;
        y = box_label[i].y;
        w = box_label[i].w;
        h = box_label[i].h;
        id = box_label[i].id;
        if ((w < lowest_w || h < lowest_h))
            ++jump;
        else {
            boxes[(i - jump) * 5 + 0] = x;
            boxes[(i - jump) * 5 + 1] = y;
            boxes[(i - jump) * 5 + 2] = w;
            boxes[(i - jump) * 5 + 3] = h;
            boxes[(i - jump) * 5 + 4] = id;
        }
    }
    
    return boxes;
}

void YOLOv3_DataLoader::correct_box(vector<Box_label> &boxes, float dx, float dy, float sx, float sy, bool flip) {
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
        
        boxes[i].x = (boxes[i].left + boxes[i].right)/2;
        boxes[i].y = (boxes[i].top + boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);
        
        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void YOLOv3_Trainer::train(int epoch) {
    //    int batch_size = network->getBatchSize();
    //    auto rng = std::mt19937((unsigned)time(NULL));
    //    vector<int> index; index.reserve(loader->size());
    //    float loss = 0;
    //    size_t data_set_size = loader->size();
    //    for (int i = 0; i < data_set_size; ++i) {
    //        index.push_back(i);
    //    }
    //    for (int i = 0; i < epoch; ++i) {
    //        printf("Epoch %d Training[", i + 1);
    //        loss = 0;
    //        shuffle(index.begin(), index.end(), rng);
    //        for (int j = 0; j + batch_size <= data_set_size; ) {
    //            Tensor data(1, 1, 416 * 416 * batch_size, 0);
    //            float *data_ptr = data.weight;
    //            vfloat label; label.reserve(5 * 90 * batch_size);
    //            for (int k = 0; k < batch_size; ++k, ++j) {
    //                Tensor src = loader->get_img(index[j]);
    ////                vfloat l = loader->get_label(index[j], 90);
    //                vfloat l;
    //                float *src_ptr = src.weight;
    //                for (int l = 0; l < src.size; ++l) {
    //                    *(data_ptr++) = *(src_ptr++);
    //                }
    //                label.insert(label.end(), l.begin(), l.end());
    //            }
    //            loss += trainer->train_batch(data, label)[0];
    //        }
    //        printf("] ");
    //        printf("loss: %f\n", loss);
    //    }
}
