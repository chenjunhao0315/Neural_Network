//
//  Image_Process.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#include "Image_Process.hpp"

IMG::~IMG() {
    
}

IMG::IMG() {
    width = 0;
    height = 0;
    channel = 0;
}

IMG::IMG(int width_, int height_, MatType type_, Scalar color) {
    width = width_;
    height = height_;
    channel = getDepth(type_);
    type = type_;
    if (!(color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 0)) {
        mat = Mat(width_, height_, type_, color);
    } else {
        mat = Mat(width_, height_, type_);
    }
}

IMG::IMG(const char *filename) {
    width = 0; height = 0;
    ImageType image_type = getType(filename);
    if (image_type == ImageType::JPEG) {
        class JPEG img(filename);
        if (img.status() != Jpeg_Status::OK) {
            if (img.status() == Jpeg_Status::UNSUPPORT) {
                printf("[JPEG] Unsupport format!\n");
            } else {
                printf("[JPEG] Decode file fail!\n");
            }
            return;
        }
        width = img.getWidth();
        height = img.getHeight();
        channel = img.getChannel();
        type = MAT_8UC3;
        if (img.getChannel() == 1)
            type = MAT_8UC1;
        mat = Mat(width, height, type, img.getPixel());
        Info = img.getPicInfo();
        img.close();
    } else if (image_type == ImageType::PPM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P6\n%d %d\n255\n", &width, &height);
        channel = 3;
        type = MAT_8UC3;
        unsigned char *pixel_array = new unsigned char [3 * width * height];
        fread(pixel_array, sizeof(unsigned char), 3 * width * height, f);
        mat = Mat(width, height, type, pixel_array);
        delete [] pixel_array;
        fclose(f);
    } else if (image_type == ImageType::PGM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P5\n%d %d\n255\n", &width, &height);
        channel = 1;
        type = MAT_8UC1;
        unsigned char *pixel_array = new unsigned char [width * height];
        fread(pixel_array, sizeof(unsigned char), width * height, f);
        mat = Mat(width, height, type, pixel_array);
        delete [] pixel_array;
        fclose(f);
    } else if (image_type == ImageType::UNSUPPORT) {
        printf("[IMG] Unsupport file type!\n");
    } else if (image_type == ImageType::OPEN_FAIL) {
        printf("[IMG] Open file fail!\n");
    }
}

IMG::IMG(const IMG &I) {
    if (this != &I) {
        width = I.width;
        height = I.height;
        channel = I.channel;
        type = I.type;
        mat = I.mat;
    }
}

IMG::IMG(IMG &&I) {
    width = I.width;
    height = I.height;
    channel = I.channel;
    type = I.type;
    mat = I.mat;
}

IMG& IMG::operator=(const IMG &I) {
    if (this != &I) {
        width = I.width;
        height = I.height;
        channel = I.channel;
        type = I.type;
        mat = I.mat;
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

unsigned char * IMG::toPixelArray() {
    return mat.ptr();
}

void IMG::showPicInfo() {
    for (int i = 0; i < Info.size(); ++i)
    printf("%s", Info[i].c_str());
}

void IMG::release() {
    Info.clear();
    mat.release();
}

void IMG::scale_channel(int channel, float scale) {
    mat.scale_channel(channel, scale);
}

void IMG::convertTo(MatType type_) {
    type = type_;
    mat = mat.convertTo(type_);
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
    IMG::ImageType store_type = phraseType(filename);
    if (store_type == IMG::ImageType::PPM) {
        if (type != MAT_8UC3) {
            printf("Channel size not correct!\n");
            return false;
        }
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(mat.ptr(), sizeof(unsigned char), 3 * width * height, f);
        fclose(f);
        return true;
    } else if (store_type == IMG::ImageType::PGM) {
        if (type != MAT_8UC1) {
            printf("Channel size not correct!\n");
            return false;
        }
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P5\n%d %d\n255\n", width, height);
        fwrite(mat.ptr(), sizeof(unsigned char), width * height, f);
        fclose(f);
        return true;
    } else if (store_type == IMG::ImageType::JPEG) {
        if (type != MAT_8UC3 && type != MAT_8UC1) {
            printf("[IMG] Unsupport data format!\n");
            return false;
        }
        class JPEG img(mat.ptr(), width, height, mat.depth());
        return img.save(filename, quality, (mat.depth() == 1) ? false : true);
    } else {
        printf("Format unsupport!\n");
    }
    return false;
}

bool IMG::empty() {
    return width == 0 && height == 0;
}

IMG IMG::resize(Size size, float factor_w, float factor_h, int method) {
    int dst_width = size.width, dst_height = size.height;
    if (size.width == 0 && size.height == 0) {
        dst_width = round(width * factor_w);
        dst_height = round(height * factor_h);
    } else {
        factor_w = (float)size.width / width;
        factor_h = (float)size.height / height;
    }
    if (type != MAT_8UC1 && type != MAT_8UC3) {
        printf("[IMG][Resize] Unsupport format\n");
        return IMG();
    }
    
    IMG result(dst_width, dst_height, type);
    
    switch(method) {
        case BILINEAR:
            bilinearResize(mat, result.getMat(), factor_w, factor_h); break;
        case NEAREST:
            nearestResize(mat, result.getMat(), factor_w, factor_h); break;
        default:
            bilinearResize(mat, result.getMat(), factor_w, factor_h);
    }
    
    return result;
}

void IMG::bilinearResize(Mat &src, Mat &dst, float factor_w, float factor_h) {
    int dst_height = dst.height;
    int dst_width = dst.width;
    int dst_step = dst.getStep();
    int dst_elemsize = dst.elemSize();
    unsigned char *dst_ptr = dst.ptr();
    
    int src_step = mat.getStep();
    int src_elemsize = mat.elemSize();
    unsigned char *src_ptr = mat.ptr();
    
    for (int i = 0; i < dst_height; ++i) {
        float index_i = (i + 0.5) / factor_h - 0.5;
        if (index_i < 0)
            index_i = 0;
        if (index_i >= height - 1)
            index_i = height - 2;
        int i_1 = int(index_i);
        index_i -= i_1;
        short int x1 = (1 - index_i) * 2048;
        short int x2 = 2048 - x1;
        for (int j = 0; j < dst_width; ++j) {
            float index_j = (j + 0.5) / factor_w - 0.5;
            if (index_j < 0)
                index_j = 0;
            if (index_j >= width - 1)
                index_j = width - 2;
            int j_1 = int(index_j);
            index_j -= j_1;
            short int y1 = (1 - index_j) * 2048;
            short int y2 = 2048 - y1;
            for (int k = 0; k < dst_elemsize; ++k) {
                *(dst_ptr + i * dst_step + j * dst_elemsize + k) = (x1 * y1 * *(src_ptr + i_1 * src_step + j_1 * src_elemsize + k) + x1 * y2 * *(src_ptr + i_1 * src_step + (j_1 + 1) * src_elemsize + k) + x2 * y1 * *(src_ptr + (i_1 + 1) * src_step + j_1 * src_elemsize + k) + x2 * y2 * *(src_ptr + (i_1 + 1) * src_step + (j_1 + 1) * src_elemsize + k)) >> 22;
            }
        }
    }
}

void IMG::nearestResize(Mat &src, Mat &dst, float factor_w, float factor_h) {
    int dst_height = dst.height;
    int dst_width = dst.width;
    int dst_step = dst.getStep();
    int dst_elemsize = dst.elemSize();
    unsigned char *dst_ptr = dst.ptr();
    
    int src_step = mat.getStep();
    int src_elemsize = mat.elemSize();
    unsigned char *src_ptr = mat.ptr();
    
    for (int i = 0; i < dst_height; ++i) {
        for (int j = 0; j < dst_width; ++j) {
            int src_h = i / factor_h;
            int src_w = j / factor_w;
            for (int k = 0; k < dst_elemsize; ++k) {
                *(dst_ptr + i * dst_step + j * dst_elemsize + k) = *(src_ptr + src_h * src_step + src_w * src_elemsize + k);
            }
        }
    }
}

IMG IMG::crop(Rect rect) {
    int x1 = rect.x1, x2 = rect.x2;
    int y1 = rect.y1, y2 = rect.y2;
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    IMG result(w, h, type);
    
    Mat &dst = result.getMat();
    int dst_elemsize = dst.elemSize();
    unsigned char *dst_ptr = dst.ptr();
    unsigned char *src_ptr = mat.ptr();
    int src_step = mat.getStep();
    int src_elemsize = mat.elemSize();
    
    unsigned char *dst_ptr_act = dst_ptr;
    unsigned char *src_ptr_act;
    
    for (int i = y1, h = 0; i <= y2; ++i, ++h) {
        for (int j = x1, w = 0; j <= x2; ++j, ++w) {
            if (i >= 0 && i < height && j >= 0 && j < width) {
                src_ptr_act = src_ptr + i * src_step + j * src_elemsize;
                for (int k = 0; k < dst_elemsize; ++k) {
                    dst_ptr_act[k] = src_ptr_act[k];
                }
            }
            dst_ptr_act += dst_elemsize;
        }
    }
    return result;
}

IMG IMG::convert(ConvertMethod method) {
    switch (method) {
        case RGB_TO_GRAY:
            return convert_rgb_to_gray();
        case RGB_TO_HSV:
            return convert_rgb_to_hsv();
        case HSV_TO_RGB:
            return convert_hsv_to_rgb();
        default:
            break;
    }
    return IMG();
}

IMG IMG::convert_rgb_to_gray() {
    if (type != MAT_8UC3) {
        printf("[IMG][RGB->GRAY] Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, MAT_8UC1);
    
    Mat &dst = result.getMat();
    unsigned char *src_ptr = mat.ptr();
    unsigned char *dst_ptr = dst.ptr();
    int size = width * height;
    
    for (int i = size; i--; ) {
        *(dst_ptr++) = 0.2127 * src_ptr[0] + 0.7152 * src_ptr[1] + 0.0722 * src_ptr[2];
        src_ptr += 3;
    }
    return result;
}

IMG IMG::convert_rgb_to_hsv() {
    if (type != MAT_8UC3) {
        printf("[IMG][RGB->HSV] Unsupport!\n");
        return IMG();
    }
    
    IMG result(width, height, MAT_32FC3);
    
    Mat &src = mat;
    Mat &dst = result.getMat();
    
    unsigned char *src_ptr = src.ptr();
    float *dst_ptr = (float*)dst.ptr();
    float scale = 1.0 / 255;
    int size = width * height;
    float r, g, b;
    float h, s, v;
    
    for (int i = size; i--; ) {
        r = src_ptr[0] * scale;
        g = src_ptr[1] * scale;
        b = src_ptr[2] * scale;
        float maximum = max(r, max(g, b));
        float minimum = min(r, min(g, b));
        float delta = maximum - minimum;
        v = maximum;
        if (maximum == 0){
            s = 0;
            h = 0;
        } else {
            s = delta / maximum;
            if(r == maximum){
                h = (g - b) / delta;
            } else if (g == maximum) {
                h = 2 + (b - r) / delta;
            } else {
                h = 4 + (r - g) / delta;
            }
            if (h < 0)
                h += 6;
            h = h / 6.0;
        }
        dst_ptr[0] = h;
        dst_ptr[1] = s;
        dst_ptr[2] = v;
        
        src_ptr += 3;
        dst_ptr += 3;
    }
    return result;
}

IMG IMG::convert_hsv_to_rgb() {
    if (type != MAT_32FC3) {
        printf("[IMG][HSV->RGB] Unsupport!\n");
        return IMG();
    }
    
    IMG result(width, height, MAT_8UC3);
    
    Mat &dst = result.getMat();
    float *src_ptr = (float*)mat.ptr();
    unsigned char *dst_ptr = dst.ptr();
    int size = width * height;
    float r, g, b;
    float h, s, v;
    float f, p, q, t;
    
    for (int i = size; i--; ) {
        h = 6 * src_ptr[0];
        s = src_ptr[1];
        v = src_ptr[2];
        if (s == 0) {
            r = g = b = v;
        } else {
            int index = floor(h);
            f = h - index;
            p = v * (1 - s);
            q = v * (1 - s * f);
            t = v * (1 - s * (1 - f));
            if(index == 0){
                r = v; g = t; b = p;
            } else if (index == 1) {
                r = q; g = v; b = p;
            } else if (index == 2) {
                r = p; g = v; b = t;
            } else if (index == 3) {
                r = p; g = q; b = v;
            } else if (index == 4) {
                r = t; g = p; b = v;
            } else {
                r = v; g = p; b = q;
            }
        }
        dst_ptr[0] = saturate_cast<unsigned char>(r * 255);
        dst_ptr[1] = saturate_cast<unsigned char>(g * 255);
        dst_ptr[2] = saturate_cast<unsigned char>(b * 255);
        
        src_ptr += 3;
        dst_ptr += 3;
    }
    
    return result;
}

IMG IMG::filter(Mat kernel, MatType dstType) {
    if (kernel.width != kernel.height) {
        printf("Unsupport!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    if (mat.depth() != getDepth(dstType)) {
        printf("[IMG][Filter] Channel unmatched!\n");
        return IMG();
    }
    
    IMG result(width, height, (dstType == MAT_UNDEFINED) ? type : dstType);
    
    Mat &dst = result.getMat();
    
    ConvTool convtool(mat.getType(), dst.getType(), kernel);
    convtool.start(mat, dst);
    
    return result;
}

IMG IMG::gaussian_blur(float radius_, float sigma_x_, float sigma_y_) {
    if (radius_ == 0 && sigma_x_ == 0) {
        printf("[IMG][Gaussian_blur] Parameter error!\n");
        return IMG();
    }
    if (type != MAT_8UC1 && type != MAT_8UC3) {
        printf("[IMG][Gaussian_blur] Unsupport type!\n");
        return IMG();
    }
    
    IMG result(width, height, type);
    int radius = (radius_ == 0) ? floor(sigma_x_ * 2.57) + 1 : (int)radius_ + 1;
    float sigma_x = (sigma_x_ == 0) ? 1.0 * radius / 2.57 : sigma_x_;
    float sigma_y = (sigma_y_ == 0) ? 1.0 * radius / 2.57 : sigma_y_;
    float normal_x = 1.0 / (sigma_x * sqrt(2.0 * PI));
    float normal_y = 1.0 / (sigma_y * sqrt(2.0 * PI));
    float coef_x = -1.0 / (2 * sigma_x * sigma_x);
    float coef_y = -1.0 / (2 * sigma_y * sigma_y);
    float filter_x[2 * (int)radius + 1];
    float filter_y[2 * (int)radius + 1];
    float gaussianSum_x = 0, gaussianSum_y = 0;
    for (int i = 0, j = -radius; j <= radius; i++, j++) {
        gaussianSum_x += filter_x[i] = normal_x * exp(1.0 * coef_x * j * j);
        gaussianSum_y += filter_y[i] = normal_y * exp(1.0 * coef_y * j * j);
    }
    for (int i = 0; i < 2 * radius + 1; i++) {
        filter_x[i] /= gaussianSum_x;
        filter_y[i] /= gaussianSum_y;
    }
    
    Mat &src = mat;
    int src_depth = src.depth();
    int src_elemsize = src.elemSize();
    int src_step = src.getStep();
    unsigned char *src_ptr = src.ptr();
    
    Mat mid(width, height, type);
    int mid_depth = mid.depth();
    int mid_elemsize = mid.elemSize();
    unsigned char *mid_ptr = mid.ptr();
    
    Mat &res = result.getMat();
    int res_depth = res.depth();
    unsigned char *res_ptr = res.ptr();
    
    int index;
    float *filter_ptr;
    float blur[3] = {0.0f};
    
    // width direction
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            filter_ptr = filter_x;
            blur[0] = 0; blur[1] = 0; blur[2] = 0;
            for (int k = -radius; k <= radius; ++k) {
                int l = w + k;
                if (l >= 0 && l < width) {
                    index = src_elemsize * l;
                    for (int d = src_depth; d--; ) {
                        blur[d] += src_ptr[index + d] * *(filter_ptr);
                    }
                }
                ++filter_ptr;
            }
            for (int d = 0; d < mid_depth; ++d) {
                *(mid_ptr++) = saturate_cast<unsigned char>(blur[d]);
            }
        }
        src_ptr += src_step;
    }
    
    mid_ptr = mid.ptr();
    
    // height direction
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            filter_ptr = filter_y;
            blur[0] = 0; blur[1] = 0; blur[2] = 0;
            for (int k = -radius; k <= radius; ++k) {
                int l = h + k;
                if (l >= 0 && l < height) {
                    index = (l * width + w) * mid_elemsize;
                    for (int d = mid_depth; d--; ) {
                        blur[d] += *(mid_ptr + index + d) * *(filter_ptr);
                    }
                }
                ++filter_ptr;
            }
            for (int d = 0; d < res_depth; ++d) {
                *(res_ptr++) = saturate_cast<unsigned char>(blur[d]);
            }
        }
    }
    return result;
}

IMG IMG::median_blur(int radius) {
    if (type != MAT_8UC1 && type != MAT_8UC3) {
        printf("[IMG][Median_blur] Unsupport type!\n");
        return IMG();
    }
    IMG dst(width, height, type);
    
    int padding = (radius - 1) / 2;
    int x, y;
    int coordinate_w, coordinate_h;
    int dst_width = width, dst_height = height, dst_channel = mat.depth();
    int src_width = width, src_channel = mat.depth();
    unsigned char *src_ptr = mat.ptr();
    unsigned char *dst_ptr = dst.getMat().ptr();
    vector<vector<unsigned char>> RGB_pixel(3, vector<unsigned char>(radius * radius));
    int src_index;
    int counter[3] = {0};
    
    y = -padding;
    for (int h = 0; h < dst_height; ++h, ++y) {
        x = -padding;
        for (int w = 0; w < dst_width; ++w, ++x) {
            counter[0] = 0; counter[1] = 0; counter[2] = 0;
            for (int kernel_h = 0; kernel_h < radius; ++kernel_h) {
                coordinate_h = y + kernel_h;
                for (int kernel_w = 0; kernel_w < radius; ++kernel_w) {
                    coordinate_w = x + kernel_w;
                    if (coordinate_w >= 0 && coordinate_w < dst_width && coordinate_h >= 0 && coordinate_h < dst_height) {
                        src_index = (coordinate_h * src_width + coordinate_w) * src_channel;
                        for (int c = 0; c < src_channel; ++c) {
                            RGB_pixel[c][counter[c]++] = (*(src_ptr + src_index + c));
                        }
                    }
                }
            }
            for (int c = 0; c < dst_channel; ++c) {
                int size = counter[c];
                if (size == 0)
                    *(dst_ptr++) = 0;
                else {
                    sort(RGB_pixel[c].begin(), RGB_pixel[c].begin() + size);
                    if (size % 2)
                        *(dst_ptr++) = (RGB_pixel[c][size >> 1] + RGB_pixel[c][(size >> 1) - 1]) / 2;
                    else
                        *(dst_ptr++) = RGB_pixel[c][size >> 1];
                }
            }
        }
    }
    return dst;
}

IMG IMG::sobel() {
    if (type != MAT_8UC1) {
        printf("[IMG][Sobel] Only accept grayscale!\n");
        return IMG();
    }
    int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
    IMG result(width, height, MAT_8UC1);
    
    int index_filter;
    unsigned char *src_ptr = mat.ptr();
    
    Mat &dst = result.getMat();
    unsigned char *dst_ptr = dst.ptr();
    
    for (int i = 0; i < height ; ++i) {
        for (int j = 0; j < width ; ++j) {
            int conv_x = 0, conv_y = 0;
            index_filter = 0;
            for (int j_h = 0; j_h < 3; ++j_h) {
                int index_h = i + j_h;
                for (int j_w = 0; j_w < 3; ++j_w) {
                    int index_w = j + j_w;
                    if (index_h >= 0 && index_h < height && index_w >= 0 && index_w < width) {
                        unsigned char src_value = *(src_ptr + index_h * width + index_w);
                        conv_x += sobel_x[index_filter] * src_value;
                        conv_y += sobel_y[index_filter] * src_value;
                    }
                    ++index_filter;
                }
            }
            *(dst_ptr++) = saturate_cast<unsigned char>(sqrt(conv_x * conv_x + conv_y * conv_y));
        }
    }
    return result;
}

IMG IMG::laplacian(float gain_) {
    Mat kernel(3, 3, MAT_32FC1);
    kernel = {0, 1, 0, 1, -4, 1, 0, 1, 0};
    IMG dst = this->filter(kernel);
    
    if (gain_ != 1) {
        Mat gain(1, 1, MAT_32FC1, Scalar(gain_));
        dst = dst.filter(gain);
    }
    
    return dst;
}

IMG IMG::canny(float threshold1, float threshold2) {
    if (mat.depth() != 1) {
        printf("[IMG][Canny] Only accept grayscale!\n");
        return IMG();
    }
    Canny CannyTool(threshold1, threshold2);
    return CannyTool.start(*this);
}

IMG IMG::threshold(unsigned char threshold, unsigned char max) {
    if (type != MAT_8UC1) {
        printf("[IMG][Threshold] Only accept grayscale!\n");
        return IMG();
    }
    IMG result(width, height, MAT_8UC1);
    
    Mat &dst = result.getMat();
    unsigned char *dst_ptr = dst.ptr();
    unsigned char *src_ptr = mat.ptr();
    
    for (int i = width * height; i--; ) {
        *(dst_ptr++) = (*(src_ptr++) > threshold) ? max : 0;
    }
    return result;
}

IMG IMG::dilate(Kernel kernel) {
    if (type != MAT_8UC1) {
        printf("[IMG][Dilate] Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("[IMG][Dilate] Kernel unsupport!\n");
        return IMG();
    }
    IMG result(width, height, MAT_8UC1);
    
    int x, y;
    int coordinate_w, coordinate_h;
    int padding = (kernel.width - 1) / 2;
    int kernel_size = kernel.width;
    int kernel_index;
    bool flag;
    
    unsigned char *src_ptr = mat.ptr();
    unsigned char *dst_ptr = result.getMat().ptr();
    
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
                        if (kernel[kernel_index] == 1 && *(src_ptr + coordinate_h * width + coordinate_w) == 255)
                            flag = true;
                    }
                }
            }
            *(dst_ptr + h * width + w) = (flag) ? 255 : 0;
        }
    }
    return result;
}

IMG IMG::erode(Kernel kernel) {
    if (type != MAT_8UC1) {
        printf("[IMG][Erode] Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("[IMG][Erode] Kernel unsupport!\n");
        return IMG();
    }
    IMG result(width, height, MAT_8UC1);
    
    int x, y;
    int coordinate_w, coordinate_h;
    int padding = (kernel.width - 1) / 2;
    int kernel_size = kernel.width;
    int kernel_index;
    bool flag;
    
    unsigned char *src_ptr = mat.ptr();
    unsigned char *dst_ptr = result.getMat().ptr();
    
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
                        if (kernel[kernel_index] == 1 && *(src_ptr + coordinate_h * width + coordinate_w) == 0)
                            flag = true;
                    }
                }
            }
            *(dst_ptr + h * width + w) = (flag) ? 0 : 255;
        }
    }
    return result;
}

IMG IMG::opening(Kernel kernel) {
    if (type != MAT_8UC1) {
        printf("[IMG][Opening] Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("[IMG][Opening] Kernel unsupport!\n");
        return IMG();
    }
    IMG result(width, height);
    result = this->erode(kernel);
    result = result.dilate(kernel);
    return result;
}

IMG IMG::closing(Kernel kernel) {
    if (type != MAT_8UC1) {
        printf("Only accept binary graph!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    IMG result(width, height);
    result = this->dilate(kernel);
    result = result.erode(kernel);
    return result;
}

IMG IMG::add(IMG &addend, MatType dstType) {
    if (mat.depth() != addend.mat.depth()) {
        printf("[IMG][Add] Channel unmatched!\n");
        return IMG();
    }
    if (!(width == addend.width && height == addend.height)) {
        printf("[IMG][Add] Size unmatched!\n");
        return IMG();
    }
    
    IMG dst_img(width, height, (dstType == MAT_UNDEFINED) ? type : dstType);
    Mat &src1 = mat;
    Mat &src2 = addend.getMat();
    Mat &dst = dst_img.getMat();
    dst = src1.add(src2, dstType);
    
    return dst_img;
}

IMG IMG::subtract(IMG &minuend, MatType dstType) {
    if (mat.depth() != minuend.mat.depth()) {
        printf("[IMG][Subtract] Channel unmatched!\n");
        return IMG();
    }
    if (!(width == minuend.width && height == minuend.height)) {
        printf("[IMG][Subtract] Size unmatched!\n");
        return IMG();
    }
    
    IMG dst_img(width, height, (dstType == MAT_UNDEFINED) ? type : dstType);
    Mat &src1 = mat;
    Mat &src2 = minuend.getMat();
    Mat &dst = dst_img.getMat();
    dst = src1.subtract(src2, dstType);
    
    return dst_img;
}

IMG IMG::scale(float scale, MatType dstType) {
    IMG dst_img = *this;
    dst_img.convertTo((dstType == MAT_UNDEFINED) ? type : dstType);
    Mat &dst_mat = dst_img.getMat();
    dst_mat = dst_mat.scale(scale);
    
    return dst_img;
}

IMG IMG::addWeighted(float alpha, IMG &addend, float beta, float gamma, MatType dstType_) {
    if (mat.depth() != addend.mat.depth()) {
        printf("[IMG][AddWeighted] Channel unmatched!\n");
        return IMG();
    }
    if (!(width == addend.width && height == addend.height)) {
        printf("[IMG][AddWeighted] Size unmatched!\n");
        return IMG();
    }
    MatType dstType = (dstType_ == MAT_UNDEFINED) ? type : dstType_;
    
    IMG dst_img(width, height, dstType);
    Mat &src1 = mat;
    Mat &src2 = addend.getMat();
    Mat &dst = dst_img.getMat();
    dst = src1.addWeighted(alpha, src2, beta, gamma, dstType);
    
    return dst_img;
}

IMG IMG::convertScaleAbs(float scale, float alpha) {
    MatType dstType = (mat.depth() == 3) ? MAT_8UC3 : MAT_8UC1;
    IMG dst_img(width, height, dstType);
    Mat &dst = dst_img.getMat();
    dst = mat.absScale(scale, alpha);
    dst = dst.convertTo(dstType);
    
    return dst_img;
}

IMG IMG::hsv_distort(float hue, float sat, float expo) {
    IMG result = this->convert(RGB_TO_HSV);
    result.scale_channel(1, sat);
    result.scale_channel(2, expo);
    float *ptr = (float*)result.getMat().ptr();
    
    for (int i = width * height; i--; ) {
        *(ptr) = *(ptr) + hue;
        if (*(ptr) > 1)
            *(ptr) -= 1;
        if (*(ptr) < 0)
            *(ptr) += 1;
        ptr += 3;
    }
    return result.convert(HSV_TO_RGB);
}

IMG IMG::local_color_correction(float radius) {
    if (type != MAT_8UC3) {
        printf("[IMG][Local Color Correction] Unsupport!\n");
        return IMG();
    }
    
    IMG mid_stage(mat.width, mat.height, MAT_8UC1);
    int size = mat.width * mat.height;
    
    unsigned char *src_ptr = mat.ptr();
    unsigned char *mid_stage_ptr = mid_stage.getMat().ptr();
    
    for (int i = size; i--; ) {
        *(mid_stage_ptr++) = 255 - ((src_ptr[0] + src_ptr[1] + src_ptr[2]) / 3);
        src_ptr += 3;
    }
    mid_stage = mid_stage.gaussian_blur(radius);
    
    IMG dst(mat.width, mat.height, MAT_8UC3);
    
    unsigned char *dst_ptr = dst.getMat().ptr();
    src_ptr = mat.ptr();
    mid_stage_ptr = mid_stage.getMat().ptr();
    
    for (int i = size; i--; ) {
        for (int j = 3; j--; ) {
            float Exp = pow(2, (128 - *(mid_stage_ptr)) / 128.0);
            int value = int(255 * pow(*(src_ptr++) / 255.0, Exp));
            *(dst_ptr++) = value;
        }
        ++mid_stage_ptr;
    }
    
    return dst;
}

void IMG::paste(IMG &img, Point p) {
    if (type != MAT_8UC3) {
        printf("[IMG][Paste] Unsupport data type!\n");
        return;
    }
    
    for (int o_h = p.y, p_h = 0; p_h < img.height; ++o_h, ++p_h) {
        for (int o_w = p.x, p_w = 0; p_w < img.width; ++o_w, ++p_w) {
            if (o_h < height && o_w < width) {
                Vec3b &dst = mat.at<Vec3b>(o_w, o_h);
                Vec3b &src = img.mat.at<Vec3b>(p_w, p_h);
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
    }
}

void IMG::flip() {
    if (type != MAT_8UC1 && type != MAT_8UC3) {
        printf("[IMG][Flip] Unsupport type!\n");
        return;
    }
    
    int row_size = width * channel;
    unsigned char *ptr;
    for (int h = 0; h < height; ++h) {
        ptr = mat.ptr<unsigned char>(h);
        for (int w = 0; w < width / 2 * channel; w += channel) {
            for (int c = 0; c < channel; ++c) {
                swap(*(ptr + w + c), *(ptr + row_size - w + c));
            }
        }
    }
}

void IMG::histogram(Size size, int resolution, const char *histogram_name) {
    if (type != MAT_8UC1 && type != MAT_8UC3) {
        printf("[IMG][Histogram] Unsupport data type!\n");
        return;
    }
    int interval = 255 / resolution;
    vector<int> calc_r(interval, 0), calc_g(interval, 0), calc_b(interval, 0);
    vector<vector<int> >calc(3, vector<int>(interval, 0));
    
    int depth = mat.depth();
    unsigned char *src_ptr = mat.ptr();
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int d = 0; d < depth; ++d) {
                calc[d][*(src_ptr++) / resolution]++;
            }
        }
    }
    
    for (int d = 0; d < depth; ++d) {
        calc[d] = normalize(calc[d], 0, size.height);
    }
    
    IMG histo(size.width, size.height, MAT_8UC3, Scalar(255, 255, 255));
    Color color_table[3] = {RED, GREEN, BLUE};
    if (depth == 1)
        color_table[0] = BLACK;
    
    float step = (float)size.width / interval;
    for (int d = 0; d < depth; ++d) {
        Color c = color_table[d];
        for (int i = 0; i < calc[d].size(); ++i) {
            histo.drawLine(Point(int(step * i), size.height - calc[d][i]), Point(int(step * (i + 1)), size.height - calc[d][i + 1]), c);
        }
    }
    
    histo.save(histogram_name, 100);
}

void IMG::drawPixel(Point p, Color color) {
    unsigned char *px = mat.ptr() + (p.y * width + p.x) * 3;
    px[0] = color.R;
    px[1] = color.G;
    px[2] = color.B;
    //    PX[p.y][p.x] = color;
}

void IMG::drawRectangle(Rect rect, Color color, int width_) {
    int l_w = (width_ == 0) ? floor(min(width, height) / 1000.0) + 1 : width_;
    int x1 = clip(rect.x1, 0 + l_w, width - l_w);
    int y1 = clip(rect.y1, 0 + l_w, height - l_w);
    int x2 = clip(rect.x2, 0 + l_w, width - l_w);
    int y2 = clip(rect.y2, 0 + l_w, height - l_w);
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

void IMG::putText(const char *str, Point p, Color color, int size) {
    Font font(size);
    size_t str_len = strlen(str);
    for (int i = 0; i < str_len; ++i) {
        Mat bitmap = font.get(str[i]);
        for (int h = p.y, y = 0; h < p.y + bitmap.height; ++h, ++y) {
            for (int w = p.x, x = 0; w < p.x + bitmap.width; ++w, ++x) {
                if (bitmap.at<unsigned char>(x, y) == 255) {
                    Vec3b &dst = mat.at<Vec3b>(w, h);
                    dst[0] = color.R;
                    dst[1] = color.G;
                    dst[2] = color.B;
                }
            }
        }
        p.x += bitmap.width;
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

IMG Canny::start(IMG &src) {
    if (threshold1 > threshold2)
        swap(threshold1, threshold2);
    
    IMG blur_img = src.gaussian_blur(3);
    
    Mat sobel_kernel_x(3, 3, MAT_32FC1);
    sobel_kernel_x = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    Mat sobel_kernel_y(3, 3, MAT_32FC1);
    sobel_kernel_y = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
    IMG sobel_x_img = blur_img.filter(sobel_kernel_x, MAT_32FC1);
    IMG sobel_y_img = blur_img.filter(sobel_kernel_y, MAT_32FC1);
    blur_img.release();
    sobel_kernel_x.release();
    sobel_kernel_y.release();
    
    Mat &sobel_x = sobel_x_img.getMat();
    Mat &sobel_y = sobel_y_img.getMat();
    Mat eta(blur_img.width, blur_img.height, MAT_32FC1, Scalar(10E-4));
    sobel_x = sobel_x.add(eta);
    eta.release();
    Mat tan = sobel_y.divide(sobel_x, MAT_32FC1);
    Mat dir = direction(tan);
    tan.release();
    sobel_x_img = sobel_x_img.convertScaleAbs();
    sobel_y_img = sobel_y_img.convertScaleAbs();
    IMG sobel = sobel_x_img.add(sobel_y_img);
    sobel_x_img.release();
    sobel_y_img.release();
    
    non_max_suppression(sobel.getMat(), dir);
    
    IMG upper = sobel.threshold(threshold2, 255);
    IMG lower = sobel.threshold(threshold1, 255);
    sobel.release();
    
    hysteresis(upper.getMat(), lower.getMat(), dir);
    
    return upper;
}

Mat Canny::direction(Mat &tan) {
    float *src = (float *)tan.ptr();
    Mat dst_mat(tan.width, tan.height, MAT_8UC1);
    unsigned char *dst = dst_mat.ptr();
    for (int i = tan.width * tan.height; i--; ) {
        float src_value = *(src++);
        if (src_value < TAN67_5 && src_value >= TAN22_5) {
            *(dst++) = DIRECTION::SLASH;
        } else if (src_value < TAN22_5 && src_value >= -TAN22_5) {
            *(dst++) = DIRECTION::HORIZONTAL;
        } else if (src_value < -TAN22_5 && src_value >= -TAN67_5) {
            *(dst++) = DIRECTION::BACK_SLASH;
        } else {
            *(dst++) = DIRECTION::VERTICAL;
        }
    }
    return dst_mat;
}

void Canny::non_max_suppression(Mat &sobel, Mat &dir) {
    int width = sobel.width;
    int height = sobel.height;
    unsigned char *dir_ptr = dir.ptr();
    unsigned char *sobel_ptr = sobel.ptr();
    
    unsigned char v1, v2;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            try {
                switch(*(dir_ptr++)) {
                    case DIRECTION::VERTICAL:
                        v1 = sobel.at<unsigned char>(i, j + 1);
                        v2 = sobel.at<unsigned char>(i, j - 1);
                        break;
                    case DIRECTION::HORIZONTAL:
                        v1 = sobel.at<unsigned char>(i + 1, j);
                        v2 = sobel.at<unsigned char>(i - 1, j);
                        break;
                    case DIRECTION::SLASH:
                        v1 = sobel.at<unsigned char>(i + 1, j + 1);
                        v2 = sobel.at<unsigned char>(i - 1, j - 1);
                        break;
                    case DIRECTION::BACK_SLASH:
                        v1 = sobel.at<unsigned char>(i + 1, j - 1);
                        v2 = sobel.at<unsigned char>(i - 1, j + 1);
                        break;
                    default:
                        v1 = 255;
                        v2 = 255;
                        break;
                }
                if (*sobel_ptr <= v1 || *sobel_ptr <= v2) {
                    *sobel_ptr = 0;
                }
            }
            catch (...) {}
            sobel_ptr++;
        }
    }
}

void Canny::hysteresis(Mat &upper, Mat &lower, Mat &dir) {
    int width = upper.width;
    int height = upper.height;
    
    Mat visited(width, height, MAT_8UC1);
    
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            if (upper.at<unsigned char>(i, j) != 0 && visited.at<unsigned char>(i, j) == 0) {
                queue<Point> q;
                q.push(Point(i, j));
                while(!q.empty()) {
                    Point p = q.front(); q.pop();
                    Point p1, p2;
                    visited.at<unsigned char>(p.x, p.y) = 255;
                    upper.at<unsigned char>(p.x, p.y) = 255;
                    unsigned char d = dir.at<unsigned char>(p.x, p.y);
                    try {
                        switch (d) {
                            case DIRECTION::VERTICAL:
                                p1 = Point(p.x + 1, p.y);
                                p2 = Point(p.x - 1, p.y);
                                break;
                            case DIRECTION::HORIZONTAL:
                                p1 = Point(p.x, p.y + 1);
                                p2 = Point(p.x, p.y - 1);
                                break;
                            case DIRECTION::SLASH:
                                p1 = Point(p.x + 1, p.y - 1);
                                p2 = Point(p.x - 1, p.y + 1);
                                break;
                            case DIRECTION::BACK_SLASH:
                                p1 = Point(p.x + 1, p.y + 1);
                                p2 = Point(p.x - 1, p.y - 1);
                                break;
                            default:
                                p1 = p;
                                p2 = p;
                                break;
                        }
                        if (lower.at<unsigned char>(p1.x, p1.y) > 0 && visited.at<unsigned char>(p1.x, p1.y) == 0)
                            q.push(p1);
                        if (lower.at<unsigned char>(p2.x, p2.y) > 0 && visited.at<unsigned char>(p2.x, p2.y) == 0)
                            q.push(p2);
                    }
                    catch (...) {}
                }
            }
        }
    }
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

Font::Font(int pixel_) {
    ascii = IMG("ascii.pgm");
    pixel = pixel_;
}

Mat Font::get(char c) {
    c -= 32;
    int w = c % 18;
    int h = c / 18;
    IMG crop = ascii.crop(Rect(w * 14, h * 18, (w + 1) * 14, (h + 1) * 18));
    crop = crop.resize(Size(0, 0), pixel / 18.0, pixel / 18.0, NEAREST);
    return crop.getMat();
}

