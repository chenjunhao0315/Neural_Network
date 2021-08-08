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

IMG::IMG(int width_, int height_, int channel_, MatType type_, Scalar color) {
    width = width_;
    height = height_;
    channel = channel_;
    type = type_;
    if (!(color.val[0] == 0 && color.val[1] == 0 && color.val[2] == 0)) {
        mat = Mat(width_, height_, type_, color);
    } else {
        mat = Mat(width_, height_, type_);
    }
}

IMG::IMG(const char *filename) {
    width = 0; height = 0; channel = 0;
    ImageType image_type = getType(filename);
    if (image_type == ImageType::JPEG) {
        class JPEG img(filename);
        if (img.status() != Jpeg_Status::OK) {
            printf("[JPEG] Decode file fail!\n");
            return;
        }
        width = img.getWidth();
        height = img.getHeight();
        channel = img.getChannel();
        type = MAT_8UC3;
        if (channel == 1)
            type = MAT_8UC1;
        mat = Mat(width, height, type, img.getPixel());
        Info = img.getPicInfo();
        img.close();
    } else if (image_type == ImageType::PPM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P6\n%d %d\n255\n", &width, &height);
        channel = 3;
        type = MAT_8UC3;
        unsigned char pixel_array[3 * width * height];
        fread(pixel_array, sizeof(unsigned char), 3 * width * height, f);
        mat = Mat(width, height, type, pixel_array);
        fclose(f);
    } else if (image_type == ImageType::PGM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P5\n%d %d\n255\n", &width, &height);
        channel = 3;
        type = MAT_8UC1;
        unsigned char pixel_array[width * height];
        fread(pixel_array, sizeof(unsigned char), width * height, f);
        mat = Mat(width, height, type, pixel_array);
        fclose(f);
    } else if (image_type == ImageType::UNSUPPORT) {
        printf("[IMG] Unsupport!\n");
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
        fwrite(mat.ptr(), sizeof(unsigned char), channel * width * height, f);
        fclose(f);
        return true;
    } else if (store_type == IMG::ImageType::PGM) {
        if (type != MAT_8UC1) {
            printf("Channel size not correct!\n");
            return false;
        }
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P5\n%d %d\n255\n", width, height);
        fwrite(mat.ptr(), sizeof(unsigned char), mat.depth() * width * height, f);
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

IMG IMG::resize(Size size, float factor_w, float factor_h) {
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
    
    IMG result(dst_width, dst_height, channel, type);
    
    Mat &dst = result.getMat();
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
    return result;
}

IMG IMG::crop(Rect rect) {
    int x1 = rect.x1, x2 = rect.x2;
    int y1 = rect.y1, y2 = rect.y2;
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    IMG result(w, h, channel, type);
    
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

IMG IMG::convertGray() {
    if (type != MAT_8UC3) {
        printf("[IMG][ConvertGray] Unsupport!\n");
        return IMG();
    }
    IMG result(width, height, 1, MAT_8UC1);
    
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

IMG IMG::filter(Mat kernel, MatType dstType) {
    if (kernel.width != kernel.height) {
        printf("Unsupport!\n");
        return IMG();
    }
    if (kernel.width % 2 == 0) {
        printf("Unsupport!\n");
        return IMG();
    }
    
    int result_channel = (dstType == MAT_UNDEFINED) ?  mat.depth() : ((dstType % 2) ? 3 : 1);
    
    IMG result(width, height, result_channel, (dstType == MAT_UNDEFINED) ? type : dstType);
    
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

    IMG result(width, height, channel, type);
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
                blur[d] = 0.0f;
            }
        }
        src_ptr += src_step;
    }
    
    mid_ptr = mid.ptr();
    
    // height direction
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            filter_ptr = filter_y;
            for (int k = -radius; k <= radius; ++k) {
                int l = h + k;
//                filter_value = *(filter_ptr++);
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
                blur[d] = 0.0f;
            }
        }
    }
    return result;
}

IMG IMG::median_blur(int radius) {
    IMG result(width, height, channel);
    
    
    
    
    
    
    return result;
}

IMG IMG::sobel() {
    if (type != MAT_8UC1) {
        printf("[IMG][Sobel] Only accept grayscale!\n");
        return IMG();
    }
    int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
    IMG result(width, height, 1, MAT_8UC1);
    
    int index_src, index_filter;
    unsigned char *src_ptr = mat.ptr();
    
    Mat &dst = result.getMat();
    unsigned char *dst_ptr = dst.ptr();
    
    for (int i = 0; i < height ; ++i) {
        for (int j = 0; j < width ; ++j) {
            int conv_x = 0, conv_y = 0;
            for (int j_h = 0; j_h < 3; ++j_h) {
                int index_h = i + j_h;
                for (int j_w = 0; j_w < 3; ++j_w) {
                    int index_w = j + j_w;
                    if (index_h >= 0 && index_h < height && index_w >= 0 && index_w < width) {
                        index_src = (index_h * width + index_w);
                        index_filter = j_h * 3 + j_w;
                        conv_x += sobel_x[index_filter] * *(src_ptr + index_src);
                        conv_y += sobel_y[index_filter] * *(src_ptr + index_src);
                    }
                }
            }
            *(dst_ptr++) = saturate_cast<unsigned char>(sqrt(conv_x * conv_x + conv_y * conv_y));
        }
    }
    return result;
}

IMG IMG::threshold(unsigned char threshold, unsigned char max) {
    if (type != MAT_8UC1) {
        printf("[IMG][Threshold] Only accept grayscale!\n");
        return IMG();
    }
    IMG result(width, height, channel, MAT_8UC1);
    
    Mat &dst = result.getMat();
    unsigned char *dst_ptr = dst.ptr();
    
    unsigned char *src_ptr = mat.ptr();
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            *(dst_ptr++) = (*(src_ptr++) > threshold) ? max : 0;
        }
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
    IMG result(width, height, channel, MAT_8UC1);
    
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
    IMG result(width, height, channel, MAT_8UC1);
    
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
    IMG result(width, height, channel);
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
    IMG result(width, height, channel);
    result = this->dilate(kernel);
    result = result.erode(kernel);
    return result;
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
    
    IMG dst_img(width, height, mat.depth(), (dstType == MAT_UNDEFINED) ? type : dstType);
    Mat &src1 = mat;
    Mat &src2 = minuend.getMat();
    Mat &dst = dst_img.getMat();
    dst = src1.subtract(src2, dstType);
    
    return dst_img;
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

    IMG histo(size.width, size.height, 3, MAT_8UC3, Scalar(255, 255, 255));
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

