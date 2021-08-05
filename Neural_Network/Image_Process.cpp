//
//  Image_Process.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#include "Image_Process.hpp"

IMG::~IMG() {
    this->freePX();
    delete [] pixel_array;
    delete [] binary_array;
}

IMG::IMG() {
    width = 0;
    height = 0;
    channel = 0;
    PX = nullptr;
    pixel_array = nullptr;
    binary_array = nullptr;
}

IMG::IMG(int width_, int height_, int channel_, Color_Space cs, Color color) {
    width = width_;
    height = height_;
    channel = channel_;
    color_space = cs;
    this->allocPX();
    pixel_array = nullptr;
    binary_array = nullptr;
    if (!(color.R == 0 && color.G == 0 && color.B == 0)) {
        fillRect(Rect(0, 0, width - 1, height - 1), color);
    }
}

IMG::IMG(const char *filename) {
    PX = nullptr;
    pixel_array = nullptr;
    binary_array = nullptr;
    width = 0; height = 0; channel = 0;
    ImageType type = getType(filename);
    if (type == ImageType::JPEG) {
        class JPEG img(filename);
        if (img.status() != Jpeg_Status::OK) {
            printf("[JPEG] Decode file fail!\n");
            return;
        }
        width = img.getWidth();
        height = img.getHeight();
        channel = img.getChannel();
        color_space = Color_Space::RGB;
        if (channel == 1)
            color_space = Color_Space::GRAY;
        this->allocPX();
        storePixelArray(img.getPixel());
        Info = img.getPicInfo();
        img.close();
    } else if (type == ImageType::PPM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P6\n%d %d\n255\n", &width, &height);
        channel = 3;
        color_space = Color_Space::RGB;
        this->allocPX();
        pixel_array = new unsigned char [3 * width * height];
        fread(pixel_array, sizeof(unsigned char), 3 * width * height, f);
        int index = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                PX[i][j].R = pixel_array[index++];
                PX[i][j].G = pixel_array[index++];
                PX[i][j].B = pixel_array[index++];
            }
        }
        fclose(f);
    } else if (type == ImageType::PGM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P5\n%d %d\n255\n", &width, &height);
        channel = 3;
        color_space = Color_Space::GRAY;
        this->allocPX();
        pixel_array = new unsigned char [width * height];
        fread(pixel_array, sizeof(unsigned char), width * height, f);
        int index = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                PX[i][j].R = pixel_array[index++];
            }
        }
        fclose(f);
    } else if (type == ImageType::UNSUPPORT) {
        printf("[IMG] Unsupport!\n");
    } else if (type == ImageType::OPEN_FAIL) {
        printf("[IMG] Open file fail!\n");
    }
}

IMG::IMG(const IMG &I) {
    PX = nullptr;
    pixel_array = nullptr;
    binary_array = nullptr;
    if (this != &I) {
        width = I.width;
        height = I.height;
        channel = I.channel;
        color_space = I.color_space;
        this->allocPX();
        this->copyPX(I.PX);
        if (I.pixel_array) {
            pixel_array = new unsigned char [width * height * channel];
            memcpy(pixel_array, I.pixel_array, sizeof(unsigned char) * channel * width * height);
        }
        if (I.binary_array) {
            binary_array = new bool [width * height];
            memcpy(binary_array, I.binary_array, sizeof(unsigned char) * width * height);
        }
    }
}

IMG::IMG(IMG &&I) {
    width = I.width;
    height = I.height;
    channel = I.channel;
    color_space = I.color_space;
    PX = I.PX;
    I.PX = nullptr;
    pixel_array = I.pixel_array;
    I.pixel_array = nullptr;
    binary_array = I.binary_array;
    I.binary_array = nullptr;
}

IMG& IMG::operator=(const IMG &I) {
    if (this != &I) {
        if (PX)
            this->freePX();
        width = I.width;
        height = I.height;
        channel = I.channel;
        color_space = I.color_space;
        this->allocPX();
        this->copyPX(I.PX);
        delete [] pixel_array;
        pixel_array = nullptr;
        if (I.pixel_array) {
            pixel_array = new unsigned char [width * height * channel];
            memcpy(pixel_array, I.pixel_array, sizeof(unsigned char) * channel * width * height);
        }
        if (I.binary_array) {
            binary_array = new bool [width * height];
            memcpy(binary_array, I.binary_array, sizeof(unsigned char) * width * height);
        }
    }
    return *this;
}

IMG::ImageType IMG::getType(const char *filename) {
    FILE *c = fopen(filename, "rb");
    if (!c)
        return ImageType::OPEN_FAIL;
    unsigned char type[3];
    fread(type, 3, 1, c);
    fclose(c);
    
    if ((type[0] == 0xFF) && (type[1] == 0xD8) && (type[2] == 0xFF)) {
        return ImageType::JPEG;
    } else if ((type[0] == 'P') && (type[1] == '6')) {
        return ImageType::PPM;
    } else if ((type[0] == 'P') && (type[1] == '5')) {
        return ImageType::PGM;
    }
    return ImageType::UNSUPPORT;
}

void IMG::allocPX() {
    PX = new PIXEL* [height];
    for (int i = 0; i < height; ++i) {
        PX[i] = new PIXEL [width];
    }
}

void IMG::copyPX(PIXEL **PX_src) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            PX[i][j] = PX_src[i][j];
        }
    }
}

void IMG::freePX() {
    if (!PX)
        return;
    for (int i = 0; i < height; ++i) {
        delete [] PX[i];
    }
    delete [] PX;
}

void IMG::storePixelArray(unsigned char *pixel_array) {
    int index = 0;
    if (channel == 1) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                PX[i][j].R = pixel_array[index++];
            }
        }
    } else {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                PX[i][j].R = pixel_array[index++];
                PX[i][j].G = pixel_array[index++];
                PX[i][j].B = pixel_array[index++];
            }
        }
    }
}

unsigned char * IMG::toPixelArray() {
    if (pixel_array)
        return pixel_array;
    if (color_space == Color_Space::BINARY) {
        pixel_array = new unsigned char [width * height];
        for (int i = 0; i < width * height; ++i) {
            pixel_array[i] = (binary_array[i]) ? 255 : 0;
        }
    } else {
        pixel_array = new unsigned char [width * height * channel];
        int index = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (channel == 1) {
                    pixel_array[index++] = PX[i][j].R;
                } else {
                    pixel_array[index++] = PX[i][j].R;
                    pixel_array[index++] = PX[i][j].G;
                    pixel_array[index++] = PX[i][j].B;
                }
            }
        }
    }
    return pixel_array;
}

void IMG::showPicInfo() {
    for (int i = 0; i < Info.size(); ++i)
        printf("%s", Info[i].c_str());
}

void IMG::release() {
    this->freePX();
    delete [] pixel_array;
    delete [] binary_array;
    Info.clear();
}

IMG::ImageType IMG::phraseType(const char *name) {
    string proc_name = string(name);
    size_t pos = proc_name.find(".");
    string type = proc_name.substr(pos + 1);
    if (type == "jpg") {
        return IMG::ImageType::JPEG;
    } else if (type == "ppm") {
        return IMG::ImageType::PPM;
    } else if (type == "pgm") {
        return IMG::ImageType::PGM;
    }
    return IMG::ImageType::UNSUPPORT;
}

bool IMG::save(const char *filename, float quality) {
    IMG::ImageType type = phraseType(filename);
    if (type == IMG::ImageType::PPM) {
        if (channel != 3) {
            printf("Channel size not correct!\n");
            return false;
        }
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        this->toPixelArray();
        fwrite(pixel_array, sizeof(unsigned char), channel * width * height, f);
        fclose(f);
        return true;
    } else if (type == IMG::ImageType::PGM) {
        if (channel != 1) {
            printf("Channel size not correct!\n");
            return false;
        }
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P5\n%d %d\n255\n", width, height);
        this->toPixelArray();
        fwrite(pixel_array, sizeof(unsigned char), channel * width * height, f);
        fclose(f);
        return true;
    } else if (type == IMG::ImageType::JPEG) {
        class JPEG img(toPixelArray(), width, height, channel);
        return img.save(filename, quality, (channel == 1) ? false : true);
    } else {
        printf("Format unsupport!\n");
    }
    return false;
}

IMG IMG::resize(Size size, float factor_w, float factor_h) {
    int dst_width = size.width, dst_height = size.height;
    if (size.width == 0 && size.height == 0) {
        dst_width = round(width * factor_w);
        dst_height = round(height * factor_h);
    } else {
        factor_w = (float)size.width / width;
        factor_h = (float)size.height / height;
    }
    
    IMG result(dst_width, dst_height, channel);
    PIXEL **pixel_src = PX;
    PIXEL **pixel_dst = result.PX;
    
    for (int i = 0; i < dst_height; ++i) {
        float index_i = (i + 0.5) / factor_h - 0.5;
        if (index_i < 0)
            index_i = 0;
        if (index_i >= height - 1)
            index_i = height - 2;
        int i_1 = int(index_i);
//        int i_2 = ceil(index_i);
        index_i -= i_1;
        int x1 = (1 - index_i) * 2048;
        int x2 = 2048 - x1;
        for (int j = 0; j < dst_width; ++j) {
            float index_j = (j + 0.5) / factor_w - 0.5;
            if (index_j < 0)
                index_j = 0;
            if (index_j >= width - 1)
                index_j = width - 2;
            int j_1 = int(index_j);
//            int j_2 = ceil(index_j);
            index_j -= j_1;
            int y1 = (1 - index_j) * 2048;
            int y2 = 2048 - y1;
//            pixel_dst[i][j].R = (1 - u) * (1 - v) * pixel_src[i_1][j_1].R + (1 - u) * v * pixel_src[i_1][j_2].R + u * (1 - v) * pixel_src[i_2][j_1].R + u * v * pixel_src[i_2][j_2].R;
//            pixel_dst[i][j].G = (1 - u) * (1 - v) * pixel_src[i_1][j_1].G + (1 - u) * v * pixel_src[i_1][j_2].G + u * (1 - v) * pixel_src[i_2][j_1].G + u * v * pixel_src[i_2][j_2].G;
//            pixel_dst[i][j].B = (1 - u) * (1 - v) * pixel_src[i_1][j_1].B + (1 - u) * v * pixel_src[i_1][j_2].B + u * (1 - v) * pixel_src[i_2][j_1].B + u * v * pixel_src[i_2][j_2].B;
            pixel_dst[i][j].R = (x1 * y1 * pixel_src[i_1][j_1].R + x1 * y2 * pixel_src[i_1][j_1 + 1].R + x2 * y1 * pixel_src[i_1 + 1][j_1].R + x2 * y2 * pixel_src[i_1 + 1][j_1 + 1].R) >> 22;
            pixel_dst[i][j].G = (x1 * y1 * pixel_src[i_1][j_1].G + x1 * y2 * pixel_src[i_1][j_1 + 1].G + x2 * y1 * pixel_src[i_1 + 1][j_1].G + x2 * y2 * pixel_src[i_1 + 1][j_1 + 1].G) >> 22;
            pixel_dst[i][j].B = (x1 * y1 * pixel_src[i_1][j_1].B + x1 * y2 * pixel_src[i_1][j_1 + 1].B + x2 * y1 * pixel_src[i_1 + 1][j_1].B + x2 * y2 * pixel_src[i_1 + 1][j_1 + 1].B) >> 22;
        }
    }
    return result;
}

IMG IMG::crop(Rect rect) {
    int x1 = rect.x1, x2 = rect.x2;
    int y1 = rect.y1, y2 = rect.y2;
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    IMG result(w, h, channel, color_space);
    
    PIXEL **pixel_dst = result.PX;
    PIXEL **pixel_src = PX;
    
    for (int i = y1, h = 0; i <= y2; ++i, ++h) {
        for (int j = x1, w = 0; j <= x2; ++j, ++w) {
            if (i >= 0 && i < height && j >= 0 && j < width) {
                pixel_dst[h][w] = pixel_src[i][j];
            }
        }
    }
    return result;
}

IMG IMG::convertGray() {
    IMG result(width, height, 1, Color_Space::GRAY);
    
    PIXEL **pixel_dst = result.PX;
    PIXEL **pixel_src = PX;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            pixel_dst[i][j].R = 0.2127 * pixel_src[i][j].R + 0.7152 * pixel_src[i][j].G + 0.0722 * pixel_src[i][j].G;
        }
    }
    return result;
}

IMG IMG::filter(int channel, Kernel kernel, bool normalize) {
    if (kernel.width != kernel.height) {
        printf("Unsupport!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    
    IMG result(width, height, channel, color_space);
    
    int padding = (kernel.width - 1) / 2;
    int kernel_size = kernel.width;
    PIXEL **pixel_src = PX;
    PIXEL **pixel_dst = result.PX;

    int x, y;
    int coordinate_w, coordinate_h;
    int conv_r, conv_g, conv_b;
    float normalization = 0;
    for (int i = 0; i < kernel_size * kernel_size; ++i)
        normalization += kernel[i];
    int kernel_index;
    
    y = -padding;
    for (int h = 0; h < height; ++h, ++y) {
        x = -padding;
        for (int w = 0; w < width; ++w, ++x) {
            conv_r = 0; conv_g = 0; conv_b = 0; kernel_index = 0;
            for (int kernel_h = 0; kernel_h < kernel_size; ++kernel_h) {
                coordinate_h = y + kernel_h;
                for (int kernel_w = 0; kernel_w < kernel_size; ++kernel_w) {
                    coordinate_w = x + kernel_w;
                    if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height) {
                        switch (channel) {
                            case 3:
                                conv_b += pixel_src[coordinate_h][coordinate_w].B * kernel[kernel_index];
                            case 2:
                                conv_g += pixel_src[coordinate_h][coordinate_w].G * kernel[kernel_index];
                            case 1:
                                conv_r += pixel_src[coordinate_h][coordinate_w].R * kernel[kernel_index];
                        }
                    }
                    kernel_index++;
                }
            }
            if (normalize && normalization) {
                switch (channel) {
                    case 3:
                        conv_b /= normalization;
                    case 2:
                        conv_g /= normalization;
                    case 1:
                        conv_r /= normalization;
                }
            }
            switch (channel) {
                case 3:
                    pixel_dst[h][w].B = clip(conv_b, 0, 255);
                case 2:
                    pixel_dst[h][w].G = clip(conv_g, 0, 255);
                case 1:
                    pixel_dst[h][w].R = clip(conv_r, 0, 255);
            }
        }
    }
    return result;
}

IMG IMG::gaussian_blur(float radius_, float sigma_x_, float sigma_y_) {
    if (radius_ == 0 && sigma_x_ == 0) {
        printf("[IMG][Gaussian_blur] Parameter error!\n");
    }
    IMG result(width, height, channel, color_space);
    IMG mid_stage(width, height, channel, color_space);
    int radius = (radius_ == 0) ? floor(sigma_x_ * 2.57) + 1 : (int)radius_ + 1;
    float *filter_x, *filter_y;
    float sigma_x = (sigma_x_ == 0) ? 1.0 * radius / 2.57 : sigma_x_;
    float sigma_y = (sigma_y_ == 0) ? 1.0 * radius / 2.57 : sigma_y_;
    float normal_x = 1.0 / (sigma_x * sqrt(2.0 * PI));
    float normal_y = 1.0 / (sigma_y * sqrt(2.0 * PI));
    float coef_x = -1.0 / (2 * sigma_x * sigma_x);
    float coef_y = -1.0 / (2 * sigma_y * sigma_y);
    filter_x = new float [2 * (int)radius + 1];
    filter_y = new float [2 * (int)radius + 1];
    float gaussianSum_x = 0, gaussianSum_y = 0;
    for (int i = 0, j = -radius; j <= radius; i++, j++) {
        gaussianSum_x += filter_x[i] = normal_x * exp(1.0 * coef_x * j * j);
        gaussianSum_y += filter_y[i] = normal_y * exp(1.0 * coef_y * j * j);
    }
    for (int i = 0; i < 2 * radius + 1; i++) {
        filter_x[i] /= gaussianSum_x;
        filter_y[i] /= gaussianSum_y;
    }
    
    float filter_value, *filter_ptr;
    // width direction
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            float blur_r = 0.0f, blur_g = 0.0f, blur_b = 0.0f;
            filter_ptr = filter_x;
            for (int k = -radius; k <= radius; ++k) {
                int l = w + k;
                filter_value = *(filter_ptr++);
                if (l >= 0 && l < width) {
                    PIXEL &PX_ACT = PX[h][l];
                    blur_r += PX_ACT.R * filter_value;
                    blur_g += PX_ACT.G * filter_value;
                    blur_b += PX_ACT.B * filter_value;
                }
            }
            PIXEL &PX_ACT = mid_stage.PX[h][w];
            PX_ACT.R = blur_r;
            PX_ACT.G = blur_g;
            PX_ACT.B = blur_b;
        }
    }
    
    // height direction
    for (int w = 0; w < width; ++w) {
        for (int h = 0; h < height; ++h) {
            float blur_r = 0.0f, blur_g = 0.0f, blur_b = 0.0f;
            filter_ptr = filter_y;
            for (int k = -radius; k <= radius; ++k) {
                int l = h + k;
                filter_value = *(filter_ptr++);
                if (l >= 0 && l < height) {
                    PIXEL &PX_ACT = mid_stage.PX[l][w];
                    blur_r += PX_ACT.R * filter_value;
                    blur_g += PX_ACT.G * filter_value;
                    blur_b += PX_ACT.B * filter_value;
                }
            }
            PIXEL &PX_ACT = result.PX[h][w];
            PX_ACT.R = blur_r;
            PX_ACT.G = blur_g;
            PX_ACT.B = blur_b;
        }
    }
    delete [] filter_x;
    delete [] filter_y;
    return result;
}

IMG IMG::median_blur(int radius) {
    IMG result(width, height, channel);
    
    
    
    
    
    
    return result;
}

IMG IMG::sobel() {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept grayscale!\n");
        return IMG();
    }
    int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    IMG result(width, height, 1, Color_Space::GRAY);
    
    for (int i = 0; i < height ; ++i) {
        for (int j = 0; j < width ; ++j) {
            int conv_x = 0, conv_y = 0;
            for (int j_h = 0; j_h < 3; ++j_h) {
                int index_h = i + j_h;
                for (int j_w = 0; j_w < 3; ++j_w) {
                    int index_w = j + j_w;
                    if (index_h >= 0 && index_h < height && index_w >= 0 && index_w < width) {
                        conv_x += sobel_x[j_h * 3 + j_h] * PX[index_h][index_w].R;
                        conv_y += sobel_y[j_h * 3 + j_h] * PX[index_h][index_w].R;
                    }
                }
            }
            result.PX[i][j].R = clip(sqrt(conv_x * conv_x + conv_y * conv_y), 0, 255);
        }
    }
    return result;
}

IMG IMG::threshold(unsigned char threshold, unsigned char max) {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept grayscale!\n");
        return IMG();
    }
    IMG result(width, height, channel, Color_Space::GRAY);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            result.PX[i][j] = (PX[i][j].R > threshold) ? max : 0;
        }
    }
    return result;
}

IMG IMG::dilate(Kernel kernel) {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, channel, Color_Space::GRAY);
    
    int x, y;
    int coordinate_w, coordinate_h;
    int padding = (kernel.width - 1) / 2;
    int kernel_size = kernel.width;
    int kernel_index;
    bool flag;
    
    y = -padding;
    for (int h = 0; h < height; ++h, ++y) {
        x = -padding;
        for (int w = 0; w < width; ++w, ++x) {
            kernel_index = 0; flag = false;
            for (int kernel_h = 0; !flag && kernel_h < kernel_size; ++kernel_h) {
                coordinate_h = y + kernel_h;
                for (int kernel_w = 0; !flag && kernel_w < kernel_size; ++kernel_w) {
                    coordinate_w = x + kernel_w;
                    if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height) {
                        if (kernel[kernel_index] == 1 && PX[coordinate_h][coordinate_w].R == 255)
                            flag = true;
                    }
                }
            }
            result.PX[h][w].R = (flag) ? 255 : 0;
        }
    }
    return result;
}

IMG IMG::erode(Kernel kernel) {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, channel, Color_Space::GRAY);
    
    int x, y;
    int coordinate_w, coordinate_h;
    int padding = (kernel.width - 1) / 2;
    int kernel_size = kernel.width;
    int kernel_index;
    bool flag;
    
    y = -padding;
    for (int h = 0; h < height; ++h, ++y) {
        x = -padding;
        for (int w = 0; w < width; ++w, ++x) {
            kernel_index = 0; flag = false;
            for (int kernel_h = 0; !flag && kernel_h < kernel_size; ++kernel_h) {
                coordinate_h = y + kernel_h;
                for (int kernel_w = 0; !flag && kernel_w < kernel_size; ++kernel_w) {
                    coordinate_w = x + kernel_w;
                    if (coordinate_w >= 0 && coordinate_w < width && coordinate_h >= 0 && coordinate_h < height) {
                        if (kernel[kernel_index] == 1 && PX[coordinate_h][coordinate_w].R == 0)
                            flag = true;
                    }
                }
            }
            result.PX[h][w].R = (flag) ? 0 : 255;
        }
    }
    return result;
}

IMG IMG::opening(Kernel kernel) {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, channel);
    result = this->erode(kernel);
    result = result.dilate(kernel);
    return result;
}

IMG IMG::closing(Kernel kernel) {
    if (color_space != Color_Space::GRAY) {
        printf("Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, channel);
    result = this->dilate(kernel);
    result = result.erode(kernel);
    return result;
}

void IMG::histogram(Size size, int resolution, const char *histogram_name) {
    int interval = 255 / resolution;
    vector<int> calc_r(interval, 0), calc_g(interval, 0), calc_b(interval, 0);
    
    PIXEL **pixel_src = PX;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            calc_r[pixel_src[i][j].R / resolution]++;
            calc_g[pixel_src[i][j].G / resolution]++;
            calc_b[pixel_src[i][j].B / resolution]++;
        }
    }
    calc_r = normalize(calc_r, 0, size.height);
    calc_g = normalize(calc_g, 0, size.height);
    calc_b = normalize(calc_b, 0, size.height);
    
    IMG histo(size.width, size.height, 3, Color_Space::RGB, Color(255, 255, 255));
    float step = (float)size.width / interval;
    if (channel == 3) {
        for (int i = 0; i < calc_r.size() - 1; ++i) {
            histo.drawLine(Point(int(step * i), size.height - calc_r[i]), Point(int(step * (i + 1)), size.height - calc_r[i + 1]), RED);
        }
        for (int i = 0; i < calc_g.size() - 1; ++i) {
            histo.drawLine(Point(step * i, size.height - calc_g[i]), Point(step * (i + 1), size.height - calc_g[i + 1]), GREEN);
        }
        for (int i = 0; i < calc_b.size() - 1; ++i) {
            histo.drawLine(Point(step * i, size.height - calc_b[i]), Point(step * (i + 1), size.height - calc_b[i + 1]), BLUE);
        }
    } else {
        for (int i = 0; i < calc_r.size() - 1; ++i) {
            histo.drawLine(Point(int(step * i), size.height - calc_r[i]), Point(int(step * (i + 1)), size.height - calc_r[i + 1]), BLACK);
        }
    }
    histo.save(histogram_name, 100);
}

void IMG::drawPixel(Point p, Color color) {
    PX[p.y][p.x] = color;
}

void IMG::drawRectangle(Rect rect, Color color, int width_) {
    int l_w = (width_ == 0) ? floor(min(width, height) / 1000.0) + 1 : width_;
    int x1 = clip(rect.x1, 0, width);
    int y1 = clip(rect.y1, 0, height);
    int x2 = clip(rect.x2, 0, width);
    int y2 = clip(rect.y2, 0, height);
    for (int x = x1; x <= x2; ++x) {
        for (int w = 0; w < l_w; ++w) {
            this->drawPixel(Point(x, y1 + w), color);
            this->drawPixel(Point(x, y2 - w), color);
        }
        
    }
    for (int y = y1; y <= y2; ++y) {
        for (int w = 0; w < l_w; ++w) {
            this->drawPixel(Point(x1 + w, y), color);
            this->drawPixel(Point(x2 - w, y), color);
        }
    }
}

// Bresenham algorithm
void IMG::drawLine(Point p1, Point p2, Color color) {
    bool steep = abs(p2.y - p1.y) > abs(p2.x - p1.x);
    if (steep) {
        swap(p1.x, p1.y);
        swap(p2.x, p2.y);
    }
    if (p1.x > p2.x) {
        swap(p1.x, p2.x);
        swap(p1.y, p2.y);
    }
    int deltax = p2.x - p1.x;
    int deltay = abs(p2.y - p1.y);
    int error = deltax / 2;
    int ystep;
    int y = p1.y;
    if (p1.y < p2.y)
        ystep = 1;
    else
        ystep = -1;
    for (int x = p1.x ; x <= p2.x; ++x) {
        if (steep)
            this->drawPixel(Point(clip(y, 0, width), clip(x, 0, height)), color);
        else
            this->drawPixel(Point(clip(x, 0, width), clip(y, 0, height)), color);
        error -= deltay;
        if (error < 0) {
            y += ystep;
            error += deltax;
        }
    }
}

void IMG::subCircle(int xc, int yc, int x, int y, Color color) {
    this->drawPixel(Point(xc + x, yc + y), color);
    this->drawPixel(Point(xc - x, yc + y), color);
    this->drawPixel(Point(xc + x, yc - y), color);
    this->drawPixel(Point(xc - x, yc - y), color);
    this->drawPixel(Point(xc + y, yc + x), color);
    this->drawPixel(Point(xc - y, yc + x), color);
    this->drawPixel(Point(xc + y, yc - x), color);
    this->drawPixel(Point(xc - y, yc - x), color);
}

void IMG::drawCircle(Point center_point, Color color, int radius_, int width_) {
    int radius = (radius_ == 0) ? (floor(min(width, height) / 250.0) + 1) : radius_;
    int radius_s = ceil(radius - (float)width_ / 2);
    int radius_e = ceil(radius + (float)width_ / 2);
    for (int r = radius_s; r < radius_e; ++r) {
        this->drawCircle_Single(center_point, color, r);
    }
}

void IMG::drawCircle_Single(Point center_point, Color color, int radius) {
    int x = 0, y = radius;
    int d = 3 - 2 * radius;
    this->subCircle(center_point.x, center_point.y, x, y, color);
    while(y >= x) {
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
        this->subCircle(center_point.x, center_point.y, x, y, color);
    }
}

void IMG::fillRect(Rect rect, Color color) {
    int x1 = clip(rect.x1, 0, width);
    int y1 = clip(rect.y1, 0, height);
    int x2 = clip(rect.x2, 0, width);
    int y2 = clip(rect.y2, 0, height);
    for (int i = x1; i <= x2; ++i) {
        for (int j = y1; j <= y2; ++j) {
            this->drawPixel(Point(i, j), color);
        }
    }
}

int clip(int value, int min, int max) {
    return value >= min ? (value < max ? value : max - 1) : min;
}

vector<int> normalize(vector<int> &src, int min, int max) {
    vector<int> normalize(src.size());
    int maximum = *max_element(src.begin(), src.end());
    int height = max - min - 1;
    float ratio = (float)height / maximum;
    for (int i = 0; i < src.size(); ++i) {
        normalize[i] = src[i] * ratio + min;
    }
    return normalize;
}

Kernel::~Kernel() {
    delete [] data;
}

Kernel::Kernel(int width_, int height_, int dimension_, float parameter) {
    width = width_;
    height = height_;
    dimension = dimension_;
    data = new float [width * height * dimension];
    fill(data, data + width * height * dimension, parameter);
}

Kernel::Kernel(const Kernel &K) {
    width = K.width;
    height = K.height;
    dimension = K.dimension;
    data = new float [width * height * dimension];
    memcpy(data, K.data, sizeof(float) * width * height * dimension);
}

Kernel::Kernel(Kernel &&K) {
    width = K.width;
    height = K.height;
    dimension = K.dimension;
    data = K.data;
    K.data = nullptr;
}

float& Kernel::operator[](int index) {
    return data[index];
}

const float& Kernel::operator[](int index) const {
    return data[index];
}

Kernel& Kernel::operator=(const Kernel &K) {
    width = K.width;
    height = K.height;
    dimension = K.dimension;
    delete [] data;
    data = new float [width * height * dimension];
    memcpy(data, K.data, sizeof(float) * width * height * dimension);
    return *this;
}

Kernel& Kernel::operator=(initializer_list<float> list) {
    int n = 0;
    for (float element : list) {
        data[n++] = element;
    }
    return *this;
}

void Kernel::show() {
    printf("Width: %d Height: %d Dimension: %d\n", width, height, dimension);
    int idx = -1;
    for (int i = 0; i < dimension; ++i) {
        printf("Dimension %d:\n", i + 1);
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                printf("%.2f ", data[++idx]);
            }
            printf("\n");
        }
    }
}

