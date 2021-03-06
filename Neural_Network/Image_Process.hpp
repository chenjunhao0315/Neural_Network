//
//  Image_Process.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#ifndef Image_Process_hpp
#define Image_Process_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <queue>

#include "Jpeg.hpp"
#include "Mat.hpp"

using namespace std;

#define PI 3.1415926

struct PIXEL {
    PIXEL(unsigned char R_ = 0, unsigned char G_ = 0, unsigned char B_ = 0) : R(R_), G(G_), B(B_) {}
    unsigned char R, G, B;
};

struct Size {
    Size(int width_ = 0, int height_ = 0) : width(width_), height(height_) {}
    int width, height;
};

struct Rect {
    Rect(int x1_, int y1_, int x2_, int y2_) : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    int x1, y1, x2, y2;
};

template <typename T>
struct Point_ {
    Point_(T x_ = -1, T y_ = -1) : x(x_), y(y_) {}
    T x, y;
};

typedef PIXEL Color;
typedef Point_<int> Point;

#define WHITE Color(255, 255, 255)
#define BLACK Color(0, 0, 0)
#define RED Color(255, 0, 0)
#define GREEN Color(0, 255, 0)
#define BLUE Color(0, 0, 255)

#define TAN22_5 0.414
#define TAN67_5 2.414

enum ResizeMethod {
    BILINEAR = 0,
    NEAREST = 1
};

enum ConvertMethod {
    SAME = 0,
    RGB_TO_GRAY,
    RGB_TO_HSV,
    HSV_TO_RGB
};

class Strel;

class IMG {
    enum ImageType {
        PPM,
        PGM,
        JPEG,
        UNSUPPORT,
        OPEN_FAIL
    };
    
    enum ColorSpace {
        GRAYSCALE,
        RGB,
        HSV
    };
public:
    ~IMG();
    IMG();
    IMG(int width, int height, MatType type = MAT_8UC3, Scalar color = Scalar(0, 0, 0));
    IMG(const char *filename);
    IMG(const IMG &I);
    IMG(IMG &&I);
    IMG& operator=(const IMG &I);
    unsigned char * toPixelArray();
    IMG resize(Size size, float factor_x = 1, float factor_y = 1, int method = 0);
    IMG crop(Rect rect, Scalar color = Scalar(0, 0, 0));
    IMG concat(IMG &src);
    IMG border(int border_width, Color color = WHITE);
    IMG convertColor(ConvertMethod method);
    IMG gaussian_blur(float radius, float sigma_x = 0, float sigma_y = 0);
    IMG median_blur(int radius);
    IMG filter(Mat kernel, MatType dstType = MAT_UNDEFINED);
    IMG sobel();
    IMG laplacian(float gain = 1);
    IMG canny(float threshold1, float threshold2);
    IMG threshold(unsigned char threshold, unsigned char max);
    IMG dilate(Strel strel);
    IMG erode(Strel strel);
    IMG opening(Strel strel);
    IMG closing(Strel strel);
    IMG add(IMG &addend, MatType dstType = MAT_UNDEFINED);
    IMG subtract(IMG &minuend, MatType dstType = MAT_UNDEFINED);
    IMG scale(float scale, MatType dstType = MAT_UNDEFINED);
    IMG addWeighted(float alpha, IMG &addend, float beta, float gamma, MatType dstType = MAT_UNDEFINED);
    IMG convertScaleAbs(float scale = 1, float alpha = 0);
    IMG hsv_distort(float hue, float sat, float expo);
    IMG local_color_correction(float radius = 10);
    Mat& getMat() {return mat;}
    void scale_channel(int channel, float scale);
    void flip();
    void convertTo(MatType type);
    void release();
    void showPicInfo();
    void paste(IMG &img, Point p);
    void histogram(Size size = Size(360, 240), int resolution = 1, const char *histogram_name = "histogram.jpg");
    void drawRectangle(Rect rect, Color color, int width = 0);
    void drawLine(Point p1, Point p2, Color color);
    void drawPixel(Point p, Color color);
    void drawCircle(Point center_point, Color color, int radius = 0, int width = 1);
    void fillRect(Rect rect, Color color);
    void putText(const char *str, Point p, Color color, int text_height = 24);
    bool save(const char *filename = "out.jpg", float quality = 80);
    bool empty();
    
    int width, height, channel;
private:
    vector<string> Info;
    MatType type;
    Mat mat;
    
    IMG::ImageType getType(const char *filename);
    IMG::ImageType phraseType(const char *name);
    void drawCircle_Single(Point center_point, Color color, int radius = 0);
    void subCircle(int xc, int yc, int x, int y, Color color);
    void bilinearResize(Mat &src, Mat &dst, float factor_w, float factor_h);
    void nearestResize(Mat &src, Mat &dst, float factor_w, float factor_h);
    IMG convert_rgb_to_gray();
    IMG convert_rgb_to_hsv();
    IMG convert_hsv_to_rgb();
};

IMG textLabel(const char *str, int text_height = 24, Color text_color = BLACK, Color background = WHITE, int border = 0);
int clip(int value, int min, int max);
vector<int> normalize(vector<int> &src, int min, int max);

class Strel {
public:
    enum Shape {
        SQUARE,
        CROSS
    };
    
    ~Strel();
    Strel(int width = 0, int height = 0, int parameter = 0);
    Strel(Strel::Shape shape, int radius);
    Strel(const Strel &K);
    Strel(Strel &&K);
    Strel& operator=(const Strel &K);
    Strel& operator=(initializer_list<int> list);
    int& operator[](int index);
    const int& operator[](int index) const;
    void show();
    
    int width;
    int height;
    int *data;
};

class Canny {
public:
    enum DIRECTION {
        SLASH,
        HORIZONTAL,
        BACK_SLASH,
        VERTICAL
    };
    Canny(float threshold1_, float threshold2_) : threshold1(threshold1_), threshold2(threshold2_) {}
    IMG start(IMG &src);
    Mat direction(Mat &tan);
    void non_max_suppression(Mat &sobel, Mat &dir);
    void hysteresis(Mat &upper, Mat &lower, Mat &dir);
private:
    float threshold1;
    float threshold2;
};

class Font {
public:
    Font(int pixel_ = 24);
    Mat getBitmap(char c);
    IMG getIMG(char c);
private:
    int pixel;
    int size;
};



#endif /* Image_Process_hpp */
