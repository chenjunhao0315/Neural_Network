//
//  Jpeg.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#ifndef Jpeg_hpp
#define Jpeg_hpp

#include <stdio.h>
#include <iostream>
#include <cstring>
#include <math.h>

#define W1 2841 /* 2048*sqrt(2)*cos(1*pi/16) */
#define W2 2676 /* 2048*sqrt(2)*cos(2*pi/16) */
#define W3 2408 /* 2048*sqrt(2)*cos(3*pi/16) */
#define W5 1609 /* 2048*sqrt(2)*cos(5*pi/16) */
#define W6 1108 /* 2048*sqrt(2)*cos(6*pi/16) */
#define W7 565  /* 2048*sqrt(2)*cos(7*pi/16) */

#define DC 0
#define AC 1

using namespace std;

enum MARKER {
    APP0_MARKER = 0xE0, APP1_MARKER = 0xE1,
    SOF0_MARKER = 0xC0, SOF2_MARKER = 0xC2,
    DQT_MARKER = 0xDB, DHT_MARKER = 0xC4,
    DRI_MARKER = 0xDD, SOS_MARKER = 0xDA
};

enum EXIF_PARAMETER {
    ExposureTime = 33434, FNumber = 33437, ExposureProgram = 34850,
    ISOSpeedRatings = 34855, SensitivityType = 34864,
    RecommendedExposureIndex = 34866, ExifVersion = 36864,
    DateTimeOriginal = 36867, CreateDate = 36868, OffsetTime = 36880,
    OffsetTimeOriginal = 36881, OffsetTimeDigitized = 36882,
    ShutterSpeedValue = 37377, ApertureValue = 37378, ExposureBiasValue = 37380,
    MaxApertureValue = 37381, MeteringMode = 37383, Flash = 37385,
    FocalLength = 37386, SubSecTime = 37520, SubSecTimeOriginal = 37521,
    SubSecTimeDigitized = 37522, FlashPixVersion = 40960, ColorSpace = 40961,
    PixelXDimension = 40962, PixelYDimension = 40963, FocalPlaneXResolution = 41486,
    FocalPlaneYResolution = 41487, FocalPlaneResolutionUnit = 41488,
    CustomRendered = 41985, ExposureMode = 41986, WhiteBalance = 41987,
    SceneCaptureType = 41990, CameraOwnerName = 42032, BodySerialNumber = 42033,
    LensSpecification = 42034, LensModel = 42036, LensSerialNumber = 42037
};

struct DATA_SET {
    enum decode_method {
        TIFF, EXIF, GPS
    };
    ~DATA_SET() {
        delete [] Tag;
        delete [] Format;
        delete [] Components;
        delete [] Offset;
        delete [] Sub;
    }
    DATA_SET(unsigned int num = 0, unsigned char *_start_pos = nullptr) {
        Init(num, _start_pos);
    }
    void Init(unsigned int num, unsigned char *_start_pos) {
        Tag = new unsigned int [num];
        Format = new unsigned int [num];
        Components = new unsigned int [num];
        Offset = new unsigned int [num];
        Sub = new unsigned int [num];
        Tag_num = num;
        method = TIFF;
        start_pos = _start_pos;
    }
    void show();
    void str(unsigned char *pos, unsigned int len);
    float ration64u(unsigned char *pos);
    float ration64s(unsigned char *pos);
    
    unsigned int Tag_num;
    unsigned char method;
    unsigned int *Tag, *Format, *Components, *Offset, *Sub;
    DATA_SET *sub;
    unsigned char *start_pos;
};

class JPEG {
public:
    ~JPEG();
    JPEG(const char *filename);
    int status() {return data.status;}
    void showPicInfo();
    void convert_ppm(const char *filename = "out.ppm");
    int getWidth();
    int getHeight();
    int getChannel();
    unsigned char * getPixel();
    
    enum Status{
        OK = 0,
        NOT_JPEG,
        SYNTAX_ERROR,
        UNSUPPORT,
        DECODE_FINISH,
        FREE
    };
    
private:
    struct VlcCode {
        unsigned char bits, code;
    };
    
    struct Component {
        unsigned char id;
        unsigned char samplex;
        unsigned char sampley;
        int width;
        int height;
        int stride;
        unsigned char quant;
        unsigned char dctable;
        unsigned char actable;
        int dc;
        unsigned char *pixels;
    };
    
    struct DATA {
        int width, height;
        unsigned char color_mode;
        Status status;
        unsigned char *data_indicator;
        unsigned char *pos;
        int size;
        int length;
        unsigned char qtab[4][64];
        Component comp[3];
        int comp_number;
        int samplemaxx, samplemaxy;
        int mcusizex, mcusizey;
        int mcuwidth, mcuheight;
        VlcCode vlctab[4][65536];
        int mcu[64];
        int buf, bufbits;
        int resetinterval;
        unsigned char *rgb;
    };
    
    inline unsigned char CF(const int x) {
        return clip((x + 64) >> 7);
    }
    
    inline void skip(int number);
    inline void GetLength();
    Status decode();
    void skipMARKER();
    void readAPP1();
    void readDataSet(DATA_SET *d, unsigned char *read_pos);
    void readDQT();
    void readSOF0();
    void readDHT();
    void readDRI();
    void readSOS();
    void readDATA();
    void decodeMCU(Component *c, unsigned char *out);
    int GetVLC(VlcCode *vlc, unsigned char *code);
    int showBits(int bits);
    void skipBits(int bits);
    int GetBits(int bits);
    void IDCTRow(int *mcu);
    void IDCTCol(const int *mcu, unsigned char *out, int stride);
    unsigned char clip(const int x);
    void toRGB();
    void upSampleV(Component *c);
    void upSampleH(Component *c);
    
    unsigned char ZigZag[64];
    DATA_SET *Exif;
    DATA data;
};

inline unsigned short Decode16(const unsigned char *pos);
inline unsigned short Get16u(const unsigned char *pos);
inline unsigned int Get32u(const unsigned char *pos);
inline int Get32s(const unsigned char *pos);

enum {
    CF4A = (-9),
    CF4B = (111),
    CF4C = (29),
    CF4D = (-3),
    CF3A = (28),
    CF3B = (109),
    CF3C = (-9),
    CF3X = (104),
    CF3Y = (27),
    CF3Z = (-3),
    CF2A = (139),
    CF2B = (-11),
};

#endif /* Jpeg_hpp */
