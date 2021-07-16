//
//  Image_Process.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#ifndef Image_Process_hpp
#define Image_Process_hpp

#include <stdio.h>
#include "Jpeg.hpp"

enum ImageType {
    PPM,
    JPEG,
    UNSUPPORT
};

struct PIXEL {
    unsigned char R, G, B;
};

struct Size {
    Size(int width_ = 0, int height_ = 0) : width(width_), height(height_) {}
    int width, height;
};

struct Rect {
    int x1, y1, x2, y2;
};

typedef PIXEL Color;

class IMG {
public:
    ~IMG();
    IMG();
    IMG(int width, int height, int channel);
    IMG(const char *filename);
    IMG(const IMG &I);
    IMG(IMG &&I);
    IMG& operator=(const IMG &I);
    void convertPPM(const char *filename);
    int getWidth();
    int getHeight();
    int getChannel();
    unsigned char * toRGB();
    IMG resize(Size size, float factor_x = 1, float factor_y = 1);
    void drawRectangle(Rect rect, Color color, int width = 0);
    void drawPixel(int x, int y, Color color);
    
    int width, height, channel;
private:
    PIXEL **PX;
    unsigned char *rgb;
    
    ImageType getType(const char *filename);
    void allocPX();
    void copyPX(PIXEL **PX_src);
    void freePX();
    void storeRGB(unsigned char *rgb);
};

int clip(int value, int min, int max);

#endif /* Image_Process_hpp */
