//
//  YOLOv3.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/8/24.
//

#include "YOLOv3.hpp"

YOLOv3::YOLOv3(int num_class_) {
    num_class = num_class_;
    network = Neural_Network();
    // Revise DarkNet53
    // Input
    network.addLayer(LayerOption{{"type", "Input"}, {"input_width", "416"}, {"input_height", "416"}, {"input_dimension", "3"}, {"name", "Input"}});
    // Conv_1 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_1"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_1"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_1"}});
    // End of Conv_1
    // Downsample Conv_2 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_2"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_2"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_2"}});
    // End of Conv_2
    // Residual Block 1
    // Conv_3(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "32"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_3"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_3"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_3"}});
    // Conv_4(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_4"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_4"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_4"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_2"}, {"name", "sc_1"}});
    // End Residual Block 1
    // Downsample Conv_5 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_5"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_5"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_5"}});
    // End Conv_5
    // Residual Block 2
    // Conv_6(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_6"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_6"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_6"}});
    // Conv_7(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_7"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_7"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_7"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_5"}, {"name", "sc_2"}});
    // End Residual Block 2
    // Residual Block 3
    // Conv_8(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "64"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_8"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_8"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_8"}});
    // Conv_9(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_9"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_9"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_9"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_2"}, {"name", "sc_3"}});
    // End Residual Block 3
    // Downsample Conv_10 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_10"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_10"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_10"}});
    // End Conv_10
    // Residual Block 4
    // Conv_11(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_11"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_11"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_11"}});
    // Conv_12(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_12"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_12"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_12"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "lr_conv_10"}, {"name", "sc_4"}});
    // End Residual Block 4
    // Residual Block 5
    // Conv_13(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_13"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_13"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_13"}});
    // Conv_14(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_14"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_14"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_14"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_4"}, {"name", "sc_5"}});
    // End Residual Block 5
    // Residual Block 6
    // Conv_15(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_15"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_15"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_15"}});
    // Conv_16(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_16"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_16"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_16"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_5"}, {"name", "sc_6"}});
    // End Residual Block 6
    // Residual Block 7
    // Conv_17(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_17"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_17"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_17"}});
    // Conv_18(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_18"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_18"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_18"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_6"}, {"name", "sc_7"}});
    // End Residual Block 7
    // Residual Block 8
    // Conv_19(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_19"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_19"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_19"}});
    // Conv_20(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_20"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_20"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_20"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_7"}, {"name", "sc_8"}});
    // End Residual Block 8
    // Residual Block 9
    // Conv_21(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_21"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_21"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_21"}});
    // Conv_22(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_22"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_22"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_22"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_8"}, {"name", "sc_9"}});
    // End Residual Block 9
    // Residual Block 10
    // Conv_23(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_23"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_23"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_23"}});
    // Conv_24(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_24"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_24"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_24"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_9"}, {"name", "sc_10"}});
    // End Residual Block 10
    // Residual Block 11
    // Conv_25(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_25"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_25"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_25"}});
    // Conv_26(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_26"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_26"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_26"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_10"}, {"name", "route_1"}});
    // End Residual Block 11
    // Extract 1
    // Downsample Conv_27 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_27"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_27"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_27"}});
    // End Conv_27
    // Residual Block 12
    // Conv_28(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_28"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_28"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_28"}});
    // Conv_29(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_29"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_29"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_29"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_11"}, {"name", "sc_12"}});
    // End Residual Block 12
    // Residual Block 13
    // Conv_30(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_30"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_30"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_30"}});
    // Conv_31(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_31"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_31"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_31"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_12"}, {"name", "sc_13"}});
    // End Residual Block 13
    // Residual Block 14
    // Conv_32(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_32"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_32"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_32"}});
    // Conv_33(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_33"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_33"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_33"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_13"}, {"name", "sc_14"}});
    // End Residual Block 14
    // Residual Block 15
    // Conv_34(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_34"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_34"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_34"}});
    // Conv_35(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_35"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_35"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_35"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_14"}, {"name", "sc_15"}});
    // End Residual Block 15
    // Residual Block 16
    // Conv_36(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_36"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_36"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_36"}});
    // Conv_37(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_37"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_37"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_37"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_15"}, {"name", "sc_16"}});
    // End Residual Block 16
    // Residual Block 17
    // Conv_38(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_38"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_38"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_38"}});
    // Conv_39(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_39"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_39"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_39"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_16"}, {"name", "sc_17"}});
    // End Residual Block 17
    // Residual Block 18
    // Conv_40(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_40"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv40"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_40"}});
    // Conv_41(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_41"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_41"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_41"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_17"}, {"name", "sc_18"}});
    // End Residual Block 18
    // Residual Block 19
    // Conv_42(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_42"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv42"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_42"}});
    // Conv_43(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_43"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_43"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_43"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_18"}, {"name", "route_2"}});
    // End Residual Block 19
    // Extract 2
    // Downsample Conv_44 with BatchNorm, LRelu
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "2"}, {"padding", "same"}, {"name", "conv_44"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_44"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_44"}});
    // End Conv_44
    // Residual Block 20
    // Conv_45(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_45"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv45"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_45"}});
    // Conv_46(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_46"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_46"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_46"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_19"}, {"name", "sc_20"}});
    // End Residual Block 20
    // Residual Block 21
    // Conv_47(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_47"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv47"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_47"}});
    // Conv_48(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_48"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_48"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_48"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_20"}, {"name", "sc_21"}});
    // End Residual Block 21
    // Residual Block 22
    // Conv_49(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_49"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv49"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_49"}});
    // Conv_50(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_50"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_50"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_50"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_21"}, {"name", "sc_22"}});
    // End Residual Block 22
    // Residual Block 23
    // Conv_51(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_51"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv51"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_51"}});
    // Conv_52(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_52"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_52"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_52"}});
    network.addLayer(LayerOption{{"type", "ShortCut"}, {"shortcut", "sc_22"}, {"name", "sc_23"}});
    // End Residual Block 23
    // End Revise DarkNet53
    
    // Conv_53(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_53"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_53"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_53"}});
    // Conv_54(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_54"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_54"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_54"}});
    // Conv_55(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_55"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_55"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_55"}});
    // Conv_56(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_56"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_56"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_56"}});
    // Conv_57(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_57"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_57"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_57"}});
    // Extract 3
    // Conv_lobj_branch(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "1024"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lobj"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_lobj"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_lobj"}});
    // End Conv_lobj_branch
    // Conv_lbbox(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (num_class + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_lbbox"}});
    // End Conv_lbbox
    
    // Conv_58(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_58"}, {"input_name", "lr_conv_57"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_58"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_58"}});
    // End Conv_58
    // UpSample
    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_1"}});
    // End UpSample
    // Concat
    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "route_2"}, {"name", "concat_1"}});
    // End Concat
    // Conv_59(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_59"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_59"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_59"}});
    // End Conv_59
    // Conv_60(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_60"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_60"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_60"}});
    // End Conv_60
    // Conv_61(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_61"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_61"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_61"}});
    // End Conv_61
    // Conv_62(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_62"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_62"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_60"}});
    // End Conv_62
    // Conv_63(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_63"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_63"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_61"}});
    // End Conv_63
    // Extract 4
    // Conv_mobj_branch(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "512"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_mobj"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_mobj"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_mobj"}});
    // End Conv_mobj_branch
    // Conv_mbbox(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (num_class + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_mbbox"}});
    // End Conv_mbbox
    
    // Conv_64(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_64"}, {"input_name", "lr_conv_61"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_64"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_64"}});
    // End Conv_64
    // UpSample
    network.addLayer(LayerOption{{"type", "UpSample"}, {"stride", "2"}, {"name", "upsample_2"}});
    // End UpSample
    // Concat
    network.addLayer(LayerOption{{"type", "Concat"}, {"concat", "route_1"}, {"name", "concat_2"}});
    // End Concat
    
    // Conv_65(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_65"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_65"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_65"}});
    // End Conv_65
    // Conv_66(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_66"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_66"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_66"}});
    // End Conv_66
    // Conv_67(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_67"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_67"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_67"}});
    // End Conv_67
    // Conv_68(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_68"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_68"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_68"}});
    // End Conv_68
    // Conv_69(1 * 1)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "128"}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_69"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_69"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_69"}});
    // End Conv_69
    // Conv_sobj_branch(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", "256"}, {"kernel_width", "3"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sobj"}});
    network.addLayer(LayerOption{{"type", "BatchNormalization"}, {"name", "bn_conv_sobj"}});
    network.addLayer(LayerOption{{"type", "LRelu"}, {"name", "lr_conv_sobj"}});
    // End Conv_sobj_branch
    // Conv_sbbox(3 * 3)
    network.addLayer(LayerOption{{"type", "Convolution"}, {"number_kernel", to_string(3 * (num_class + 5))}, {"kernel_width", "1"}, {"stride", "1"}, {"padding", "same"}, {"name", "conv_sbbox"}});
    // End Conv_sbbox
    
    
    network.makeLayer(1);
    network.shape();
    network.save("test.bin");
}
