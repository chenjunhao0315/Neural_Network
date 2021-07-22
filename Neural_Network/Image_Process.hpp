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

typedef vector<vector<int >> Mat;

class IMG {
    enum ImageType {
        PPM,
        JPEG,
        UNSUPPORT
    };
public:
    ~IMG();
    IMG();
    IMG(int width, int height, int channel, bool isRGB = true);
    IMG(const char *filename);
    IMG(const IMG &I);
    IMG(IMG &&I);
    IMG& operator=(const IMG &I);
    void convertPPM(const char *filename);
    unsigned char * toRGB();
    IMG resize(Size size, float factor_x = 1, float factor_y = 1);
    IMG crop(Rect rect);
    IMG guassian_blur(float radius);
    IMG filter(int channel, Mat kernel);
    void drawRectangle(Rect rect, Color color, int width = 0);
    void drawLine(Point p1, Point p2, Color color);
    void drawPixel(Point p, Color color);
    void drawCircle(Point center_point, int radius, Color color);
    void save(const char *filename = "out.jpg", float quality = 100);
    
    int width, height, channel;
private:
    PIXEL **PX;
    unsigned char *rgb;
    bool isRGB;
    
    IMG::ImageType getType(const char *filename);
    IMG::ImageType phraseType(const char *name);
    void allocPX();
    void copyPX(PIXEL **PX_src);
    void freePX();
    void storeRGB(unsigned char *rgb);
    void subCircle(int xc, int yc, int x, int y, Color color);
};

int clip(int value, int min, int max);

#endif /* Image_Process_hpp */
