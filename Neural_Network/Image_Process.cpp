//
//  Image_Process.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#include "Image_Process.hpp"

IMG::~IMG() {
    this->freePX();
    delete [] rgb;
}

IMG::IMG() {
    PX = nullptr;
    rgb = nullptr;
}

IMG::IMG(int width_, int height_, int channel_, bool isRGB) {
    width = width_;
    height = height_;
    channel = channel_;
    this->allocPX();
    rgb = nullptr;
}

IMG::IMG(const char *filename) {
    PX = nullptr;
    rgb = nullptr;
    ImageType type = getType(filename);
    if (type == ImageType::JPEG) {
        class JPEG img(filename);
        if (img.status() == JPEG::Status::NOT_JPEG) {
            printf("Open file fail!\n");
            return;
        }
        width = img.getWidth();
        height = img.getHeight();
        channel = img.getChannel();
        this->allocPX();
        storeRGB(img.getPixel());
    } else if (type == ImageType::PPM) {
        FILE *f = fopen(filename, "r");
        fscanf(f, "P6\n%d %d\n255\n", &width, &height);
        channel = 3;
        this->allocPX();
        rgb = new unsigned char [3 * width * height];
        fread(rgb, sizeof(unsigned char), 3 * width * height, f);
        int index = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                PX[i][j].R = rgb[index++];
                PX[i][j].G = rgb[index++];
                PX[i][j].B = rgb[index++];
            }
        }
        fclose(f);
    }
}

IMG::IMG(const IMG &I) {
    PX = nullptr;
    rgb = nullptr;
    if (this != &I) {
        width = I.width;
        height = I.height;
        channel = I.channel;
        this->allocPX();
        this->copyPX(I.PX);
        if (I.rgb) {
            rgb = new unsigned char [width * height * channel];
            memcpy(rgb, I.rgb, sizeof(unsigned char) * channel * width * height);
        }
    }
}

IMG::IMG(IMG &&I) {
    width = I.width;
    height = I.height;
    channel = I.channel;
    PX = I.PX;
    I.PX = nullptr;
    rgb = I.rgb;
    I.rgb = nullptr;
}

IMG& IMG::operator=(const IMG &I) {
    PX = nullptr;
    rgb = nullptr;
    if (this != &I) {
        if (PX)
            this->freePX();
        width = I.width;
        height = I.height;
        channel = I.channel;
        this->allocPX();
        this->copyPX(I.PX);
        delete [] rgb;
        rgb = nullptr;
        if (I.rgb) {
            rgb = new unsigned char [width * height * channel];
            memcpy(rgb, I.rgb, sizeof(unsigned char) * channel * width * height);
        }
    }
    return *this;
}

IMG::ImageType IMG::getType(const char *filename) {
    FILE *c = fopen(filename, "rb");
    if (!c)
        return ImageType::UNSUPPORT;
    unsigned char type[2];
    fread(type, 2, 1, c);
    fclose(c);
    
    if ((type[0] == 0xFF) && (type[1] == 0xD8)) {
        return ImageType::JPEG;
    } else if ((type[0] == 'P') && (type[1] == '6')) {
        return ImageType::PPM;
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
    for (int i = 0; i < height; ++i) {
        delete [] PX[i];
    }
    delete [] PX;
}

void IMG::storeRGB(unsigned char *rgb) {
    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            PX[i][j].R = rgb[index++];
            PX[i][j].G = rgb[index++];
            PX[i][j].B = rgb[index++];
        }
    }
}

unsigned char * IMG::toRGB() {
    if (rgb)
        return rgb;
    rgb = new unsigned char [width * height * channel];
    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            rgb[index++] = PX[i][j].R;
            rgb[index++] = PX[i][j].G;
            rgb[index++] = PX[i][j].B;
        }
    }
    return rgb;
}

void IMG::convertPPM(const char *filename) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    this->toRGB();
    fwrite(rgb, sizeof(unsigned char), channel * width * height, f);
    fclose(f);
}

IMG::ImageType IMG::phraseType(const char *name) {
    string proc_name = string(name);
    size_t pos = proc_name.find(".");
    string type = proc_name.substr(pos + 1);
    if (type == "jpg") {
        return IMG::ImageType::JPEG;
    } else if (type == "ppm") {
        return IMG::ImageType::PPM;
    }
    return IMG::ImageType::UNSUPPORT;
}

void IMG::save(const char *filename, float quality) {
    IMG::ImageType type = phraseType(filename);
    if (type == IMG::ImageType::PPM) {
        FILE *f = fopen(filename, "wb");
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        this->toRGB();
        fwrite(rgb, sizeof(unsigned char), channel * width * height, f);
        fclose(f);
    } else if (type == IMG::ImageType::JPEG) {
        class JPEG img(toRGB(), width, height, channel);
        img.save(filename, quality);
    } else {
        printf("Format unsupport!\n");
    }
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
    
    for (int i = 0; i < dst_height; ++i) {
        float index_i = (i + 0.5) / factor_h - 0.5;
        if (index_i < 0)
            index_i = 0;
        if (index_i >= height - 1)
            index_i = height - 2;
        int i_1 = floor(index_i);
        int i_2 = ceil(index_i);
        float u = index_i - i_1;
        for (int j = 0; j < dst_width; ++j) {
            float index_j = (j + 0.5) / factor_w - 0.5;
            if (index_j < 0)
                index_j = 0;
            if (index_j >= width - 1)
                index_j = width - 2;
            int j_1 = floor(index_j);
            int j_2 = ceil(index_j);
            float v = index_j - j_1;
            result.PX[i][j].R = (1 - u) * (1 - v) * PX[i_1][j_1].R + (1 - u) * v * PX[i_1][j_2].R + u * (1 - v) * PX[i_2][j_1].R + u * v * PX[i_2][j_2].R;
            result.PX[i][j].G = (1 - u) * (1 - v) * PX[i_1][j_1].G + (1 - u) * v * PX[i_1][j_2].G + u * (1 - v) * PX[i_2][j_1].G + u * v * PX[i_2][j_2].G;
            result.PX[i][j].B = (1 - u) * (1 - v) * PX[i_1][j_1].B + (1 - u) * v * PX[i_1][j_2].B + u * (1 - v) * PX[i_2][j_1].B + u * v * PX[i_2][j_2].B;
        }
    }
    return result;
}

IMG IMG::crop(Rect rect) {
    int w = rect.x2 - rect.x1 + 1;
    int h = rect.y2 - rect.y1 + 1;
    IMG result(w, h, channel);
    
    for (int i = rect.y1, h = 0; i <= rect.y2; ++i, ++h) {
        for (int j = rect.x1, w = 0; j <= rect.x2; ++j, ++w) {
            if (i >= 0 && i < height && j >= 0 && j < width) {
                result.PX[h][w] = PX[i][j];
            }
        }
    }
    return result;
}

IMG IMG::filter(int channel, Mat kernel) {
    if (kernel.size() != kernel[0].size()) {
        printf("Unsupport!\n");
        return IMG();
    }
    int padding = kernel.size();
    return IMG();
}

IMG IMG::guassian_blur(float radius) {
    return IMG();
}

void IMG::drawPixel(Point p, Color color) {
    PX[p.y][p.x] = color;
}

void IMG::drawRectangle(Rect rect, Color color, int width_) {
    int l_w = (width_ == 0) ? floor(min(width, height)) / 1000 + 1 : width_;
    int x1 = clip(rect.x1, 0, width);
    int y1 = clip(rect.y1, 0, height);
    int x2 = clip(rect.x2, 0, width);
    int y2 = clip(rect.y2, 0, height);
    for (int x = x1; x <= x2; ++x) {
        for (int w = 0; w < l_w; ++w) {
            drawPixel(Point(x, y1 + w), color);
            drawPixel(Point(x, y2 - w), color);
        }
        
    }
    for (int y = y1; y <= y2; ++y) {
        for (int w = 0; w < l_w; ++w) {
            drawPixel(Point(x1 + w, y), color);
            drawPixel(Point(x2 - w, y), color);
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
            drawPixel(Point(clip(y, 0, height), clip(x, 0, width)), color);
        else
            drawPixel(Point(clip(x, 0, width), clip(y, 0, height)), color);
        error -= deltay;
        if (error < 0) {
            y += ystep;
            error += deltax;
        }
    }
}

void IMG::subCircle(int xc, int yc, int x, int y, Color color) {
    drawPixel(Point(xc + x, yc + y), color);
    drawPixel(Point(xc - x, yc + y), color);
    drawPixel(Point(xc + x, yc - y), color);
    drawPixel(Point(xc - x, yc - y), color);
    drawPixel(Point(xc + y, yc + x), color);
    drawPixel(Point(xc - y, yc + x), color);
    drawPixel(Point(xc + y, yc - x), color);
    drawPixel(Point(xc - y, yc - x), color);
}

void IMG::drawCircle(Point center_point, int radius, Color color) {
    int x = 0, y = radius;
    int d = 3 - 2 * radius;
    subCircle(center_point.x, center_point.y, x, y, color);
    while(y >= x) {
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
        subCircle(center_point.x, center_point.y, x, y, color);
    }
}

int clip(int value, int min, int max) {
    return value >= min ? (value < max ? value : max - 1) : min;
}
