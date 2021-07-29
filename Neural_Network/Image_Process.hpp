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
#include "Jpeg.hpp"

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

struct Point {
    Point(int x_, int y_) : x(x_), y(y_) {}
    int x, y;
};

typedef PIXEL Color;

#define RED Color(255, 0, 0)
#define GREEN Color(0, 255, 0)
#define BLUE Color(0, 0, 255)

typedef float* Mat;

class IMG {
    enum ImageType {
        PPM,
        PGM,
        JPEG,
        UNSUPPORT
    };
public:
    ~IMG();
    IMG();
    IMG(int width, int height, int channel, Color color = Color(0, 0, 0));
    IMG(const char *filename);
    IMG(const IMG &I);
    IMG(IMG &&I);
    IMG& operator=(const IMG &I);
    unsigned char * toPixelArray();
    IMG resize(Size size, float factor_x = 1, float factor_y = 1);
    IMG crop(Rect rect);
    IMG convertGray();
    IMG gaussian_blur(float radius, float sigma_x = -1, float sigma_y = -1);
    IMG median_blur(int radius);
    IMG filter(int channel, Mat kernel, Size kernel_size, bool normalize = true);
    IMG sobel();
    void histogram(Size size = Size(360, 240), int resolution = 1, const char *histogram_name = "histogram.jpg");
    void drawRectangle(Rect rect, Color color, int width = 0);
    void drawLine(Point p1, Point p2, Color color);
    void drawPixel(Point p, Color color);
    void drawCircle(Point center_point, Color color, int radius = 0);
    void fillRect(Rect rect, Color color);
    bool save(const char *filename = "out.jpg", float quality = 80);
    
    int width, height, channel;
private:
    PIXEL **PX;
    unsigned char *pixel_array;
    bool isRGB;
    
    IMG::ImageType getType(const char *filename);
    IMG::ImageType phraseType(const char *name);
    void allocPX();
    void copyPX(PIXEL **PX_src);
    void freePX();
    void storePixelArray(unsigned char *rgb);
    void subCircle(int xc, int yc, int x, int y, Color color);
};

int clip(int value, int min, int max);
vector<int> normalize(vector<int> &src, int min, int max);

#endif /* Image_Process_hpp */
