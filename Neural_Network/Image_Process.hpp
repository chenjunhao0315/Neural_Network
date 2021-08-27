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

class Kernel;

class IMG {
    enum ImageType {
        PPM,
        PGM,
        JPEG,
        UNSUPPORT,
        OPEN_FAIL
    };
public:
    ~IMG();
    IMG();
    IMG(int width, int height, int channel, MatType type = MAT_8UC3, Scalar color = Scalar(0, 0, 0));
    IMG(const char *filename);
    IMG(const IMG &I);
    IMG(IMG &&I);
    IMG& operator=(const IMG &I);
    unsigned char * toPixelArray();
    IMG resize(Size size, float factor_x = 1, float factor_y = 1, int method = 0);
    IMG crop(Rect rect);
    IMG convertGray();
    IMG gaussian_blur(float radius, float sigma_x = 0, float sigma_y = 0);
    IMG median_blur(int radius);
    IMG filter(Mat kernel, MatType dstType = MAT_UNDEFINED);
    IMG sobel();
    IMG laplacian(float gain = 1);
    IMG canny(float threshold1, float threshold2);
    IMG threshold(unsigned char threshold, unsigned char max);
    IMG dilate(Kernel kernel);
    IMG erode(Kernel kernel);
    IMG opening(Kernel kernel);
    IMG closing(Kernel kernel);
    IMG add(IMG &addend, MatType dstType = MAT_UNDEFINED);
    IMG subtract(IMG &minuend, MatType dstType = MAT_UNDEFINED);
    IMG addWeighted(float alpha, IMG &addend, float beta, float gamma, MatType dstType = MAT_UNDEFINED);
    IMG convertScaleAbs(float scale = 1, float alpha = 0);
    IMG local_color_correction(float radius = 10);
    Mat& getMat() {return mat;}
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
    void putText(const char *str, Point p, Color color, int size = 18);
    bool save(const char *filename = "out.jpg", float quality = 80);
    
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
};

int clip(int value, int min, int max);
vector<int> normalize(vector<int> &src, int min, int max);

class Kernel {
public:
    ~Kernel();
    Kernel(int width = 0, int height = 0, int dimension = 0, float parameter = 0);
    Kernel(const Kernel &K);
    Kernel(Kernel &&K);
    Kernel& operator=(const Kernel &K);
    Kernel& operator=(initializer_list<float> list);
    float& operator[](int index);
    const float& operator[](int index) const;
    void show();
    
    int width;
    int height;
    int dimension;
    float *data;
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
    Font(int pixel_ = 18);
    Mat get(char c);
private:
    IMG ascii;
    int pixel;
};



#endif /* Image_Process_hpp */
