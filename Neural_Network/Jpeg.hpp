//
//  Jpeg.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#ifndef Jpeg_hpp
#define Jpeg_hpp

#define W1 2841 /* 2048*sqrt(2)*cos(1*pi/16) */
#define W2 2676 /* 2048*sqrt(2)*cos(2*pi/16) */
#define W3 2408 /* 2048*sqrt(2)*cos(3*pi/16) */
#define W5 1609 /* 2048*sqrt(2)*cos(5*pi/16) */
#define W6 1108 /* 2048*sqrt(2)*cos(6*pi/16) */
#define W7 565  /* 2048*sqrt(2)*cos(7*pi/16) */

#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>
#include <math.h>
#include <vector>

using namespace std;

class JPEG;
class JPEG_ENCODER;
class JPEG_DECODER;
class BIT_WRITER;
class JFIF;

enum Jpeg_Status {
    OK = 0,
    NOT_JPEG,
    SYNTAX_ERROR,
    UNSUPPORT,
    DECODE_FINISH,
    DECODED_MODE,
    FREE
};

struct BitCode {
    BitCode() = default;
    BitCode(unsigned short code_, unsigned short numBits_) : code(code_), numBits(numBits_) {}
    unsigned short code;
    unsigned char numBits;
};

class JPEG {
public:
    ~JPEG();
    JPEG(const char *filename);
    JPEG(unsigned char *pixelArray, int width, int height, int channel);
    int status() {return decode_status;}
    bool save(const char *filename = "out.jpg", float quality = 90, bool isRGB = true, bool down_sample = false);
    int getWidth() {return width;}
    int getHeight() {return height;}
    int getChannel() {return channel;}
    unsigned char * getPixel() {return pixelArray;}
    
    int width;
    int height;
    int channel;
private:
    unsigned char *pixelArray;
    JPEG_DECODER *decoder;
    JPEG_ENCODER *encoder;
    vector<string> Info;
    Jpeg_Status decode_status;
};

class JPEG_DECODER {
public:
    JPEG_DECODER(const char *filename);
    
    Jpeg_Status status() {return data.status;}
    void getPicInfo(vector<string> &Info);
    
    int getWidth() {return data.width;}
    int getHeight() {return data.height;}
    int getChannel() {return data.comp_number;}
    unsigned char * getPixel();
private:
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
    
    struct Data {
        int width, height;
        unsigned char color_mode;
        Jpeg_Status status;
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
        BitCode vlctab[4][65536];
        int mcu[64];
        int buf, bufbits;
        int resetinterval;
        unsigned char *rgb;
    };
    
    inline void skip(int number);
    inline void GetLength();
    
    Jpeg_Status decode();
    void skipMARKER();
    void readAPP1();
    void readJFIF(JFIF *d, unsigned char *read_pos);
    void readDQT();
    void readSOF0();
    void readDHT();
    void readDRI();
    void readSOS();
    void readDATA();
    void decodeMCU(Component *c, unsigned char *out);
    
    int GetVLC(BitCode *vlc, unsigned char *code);
    int showBits(int bits);
    void skipBits(int bits);
    int GetBits(int bits);
    
    void IDCTRow(int *mcu);
    void IDCTCol(const int *mcu, unsigned char *out, int stride);
    unsigned char clip(const int x);
    inline unsigned char CF(const int x);
    void toRGB();
    void upSampleV(Component *c);
    void upSampleH(Component *c);
    
    JFIF *Exif;
    Data data;
};

struct JFIF {
//public:
    ~JFIF();
    JFIF(unsigned int num = 0, unsigned char *_start_pos = nullptr) {
        Init(num, _start_pos);
    }
    void Init(unsigned int num, unsigned char *_start_pos);
    void getInfo(vector<string> &Info);
    
//private:
    enum decode_method {
        TIFF, EXIF, GPS
    };
    
    void str(unsigned char *pos, unsigned int len);
    float ration64u(unsigned char *pos);
    float ration64s(unsigned char *pos);
    
    unsigned int Tag_num;
    unsigned char method;
    unsigned int *Tag, *Format, *Components, *Offset, *Sub;
    JFIF *sub;
    unsigned char *start_pos;
};

class JPEG_ENCODER {
public:
    JPEG_ENCODER(unsigned char *pixelArray_, int width_, int height_, int channel_) : pixelArray(pixelArray_), width(width_), height(height_), channel(channel_) {}
    bool write(const char *filename, float quality_, bool down_sample);
private:
    void generateHuffmanTable(const unsigned char numCodes[16], const unsigned char *values, BitCode result[256]);
    float convertRGBtoY(float r, float g, float b);
    float convertRGBtoCb(float r, float g, float b);
    float convertRGBtoCr(float r, float g, float b);
    int encodeBlock(BIT_WRITER& writer, float block[8][8], const float scaled[64], int lastDC, const BitCode huffmanDC[256], const BitCode huffmanAC[256], const BitCode* codewords);
    void DCT(float block[8*8], unsigned short stride);
    
    int width;
    int height;
    int channel;
    unsigned char *pixelArray;
};

inline unsigned short Decode16(const unsigned char *pos);
inline unsigned short Get16u(const unsigned char *pos);
inline unsigned int Get32u(const unsigned char *pos);
inline int Get32s(const unsigned char *pos);

class BIT_WRITER {
public:
    ~BIT_WRITER();
    BIT_WRITER(const char *filename);
    void write(const void *data, int size);
    void write_byte(unsigned char data);
    void write_word(unsigned short data);
    void write_bits(const BitCode &data);
    void addMarker(unsigned char id, unsigned short length);
    void flush();
private:
    struct BitBuffer {
        int data = 0;
        unsigned char numBits = 0;
    } buffer;
    FILE *f;
};

template <typename Number, typename Limit>
Number clamp(Number value, Limit minValue, Limit maxValue);

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

const char ZigZagInv[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

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

const unsigned char DefaultQuantLuminance[8 * 8] =
{ 16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99 };

const unsigned char DefaultQuantChrominance[8 * 8] =
{ 17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99 };

const unsigned char DcLuminanceCodesPerBitsize[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

const unsigned char DcLuminanceValues[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

const unsigned char AcLuminanceCodesPerBitsize[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125};

const unsigned char AcLuminanceValues[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13,
    0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
    0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A,
    0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35,
    0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
    0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84,
    0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3,
    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
    0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA
};

const unsigned char DcChrominanceCodesPerBitsize[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

const unsigned char DcChrominanceValues[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

const unsigned char AcChrominanceCodesPerBitsize[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119};

const unsigned char AcChrominanceValues[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51,
    0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1,
    0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24,
    0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A,
    0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66,
    0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82,
    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA,
    0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
    0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA
};

const int CodeWordLimit = 2048;

#endif /* Jpeg_hpp */
