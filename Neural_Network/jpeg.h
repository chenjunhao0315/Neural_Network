//
//  jpeg.h
//  jpeg_encoder_decoder
//
//  Created by 陳均豪 on 2021/5/26.
//

#ifndef jpeg_h
#define jpeg_h

#define W1 2841 /* 2048*sqrt(2)*cos(1*pi/16) */
#define W2 2676 /* 2048*sqrt(2)*cos(2*pi/16) */
#define W3 2408 /* 2048*sqrt(2)*cos(3*pi/16) */
#define W5 1609 /* 2048*sqrt(2)*cos(5*pi/16) */
#define W6 1108 /* 2048*sqrt(2)*cos(6*pi/16) */
#define W7 565  /* 2048*sqrt(2)*cos(7*pi/16) */

#define DC 0
#define AC 1

#include <cstring>

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
        delete start_pos;
        delete sub;
    }
    DATA_SET(unsigned int num = 0, unsigned char *_start_pos = 0) {
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
    void convert();
    
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

JPEG::~JPEG() {
    if (Exif)
        delete Exif;
    if (data.data_indicator)
        delete [] data.data_indicator;
    if (data.comp[0].pixels)
        delete [] data.comp[0].pixels;
    if (data.comp[1].pixels)
        delete [] data.comp[1].pixels;
    if (data.comp[2].pixels)
        delete [] data.comp[2].pixels;
    if (data.rgb)
        delete [] data.rgb;
}

JPEG::JPEG(const char *filename) {
    FILE *f = fopen(filename, "rb");
    size_t size;
    if (!f) {
        printf("Error opening the input file!\n");
        return;
    }
    memset(&data, 0, sizeof(data));
    data.comp[0].pixels = nullptr;
    data.comp[1].pixels = nullptr;
    data.comp[2].pixels = nullptr;
    data.pos = nullptr;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    data.data_indicator = data.pos = new unsigned char [size];
    if (!data.pos) {
        printf("Error allocating memory!\n");
    }
    data.size = size & 0x7FFFFFFF;
    data.status = OK;
    fseek(f, 0, SEEK_SET);
    fread(data.pos, 1, size, f);
    fclose(f);
    char table[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
        11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
        42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
        38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
    memcpy(ZigZag, table, sizeof(ZigZag));
    data.status = decode();
    switch (data.status) {
        case OK: printf("Decode finish!\n"); break;
        case NOT_JPEG: printf("Not jpeg file!\n"); break;
        case SYNTAX_ERROR: printf("Syntax error!\n"); break;
        case UNSUPPORT: printf("Unsupport!\n"); break;
        default: break;
    }
}

void JPEG::convert() {
    if (status() != DECODE_FINISH)
        return;
    FILE *f = fopen("out.ppm", "wb");
    fprintf(f, "P%d\n%d %d\n255\n", data.comp_number == 3 ? 6 : 5, data.width, data.height);
    fwrite(data.rgb, 1, data.width * data.height * data.comp_number, f);
    fclose(f);
}

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

void JPEG::upSampleH(Component* c) {
    const int xmax = c->width - 3;
    unsigned char *out, *lin, *lout;
    int x, y;
    out = new unsigned char [(c->width * c->height) << 1];
    lin = c->pixels;
    lout = out;
    for (y = c->height;  y;  --y) {
        lout[0] = CF(CF2A * lin[0] + CF2B * lin[1]);
        lout[1] = CF(CF3X * lin[0] + CF3Y * lin[1] + CF3Z * lin[2]);
        lout[2] = CF(CF3A * lin[0] + CF3B * lin[1] + CF3C * lin[2]);
        for (x = 0;  x < xmax;  ++x) {
            lout[(x << 1) + 3] = CF(CF4A * lin[x] + CF4B * lin[x + 1] + CF4C * lin[x + 2] + CF4D * lin[x + 3]);
            lout[(x << 1) + 4] = CF(CF4D * lin[x] + CF4C * lin[x + 1] + CF4B * lin[x + 2] + CF4A * lin[x + 3]);
        }
        lin += c->stride;
        lout += c->width << 1;
        lout[-3] = CF(CF3A * lin[-1] + CF3B * lin[-2] + CF3C * lin[-3]);
        lout[-2] = CF(CF3X * lin[-1] + CF3Y * lin[-2] + CF3Z * lin[-3]);
        lout[-1] = CF(CF2A * lin[-1] + CF2B * lin[-2]);
    }
    c->width <<= 1;
    c->stride = c->width;
    delete [] c->pixels;
    c->pixels = out;
}

void JPEG::upSampleV(Component* c) {
    const int w = c->width, s1 = c->stride, s2 = s1 + s1;
    unsigned char *out, *cin, *cout;
    int x, y;
    out = new unsigned char [(c->width * c->height) << 1];
    for (x = 0;  x < w;  ++x) {
        cin = &c->pixels[x];
        cout = &out[x];
        *cout = CF(CF2A * cin[0] + CF2B * cin[s1]);  cout += w;
        *cout = CF(CF3X * cin[0] + CF3Y * cin[s1] + CF3Z * cin[s2]);  cout += w;
        *cout = CF(CF3A * cin[0] + CF3B * cin[s1] + CF3C * cin[s2]);  cout += w;
        cin += s1;
        for (y = c->height - 3;  y;  --y) {
            *cout = CF(CF4A * cin[-s1] + CF4B * cin[0] + CF4C * cin[s1] + CF4D * cin[s2]);  cout += w;
            *cout = CF(CF4D * cin[-s1] + CF4C * cin[0] + CF4B * cin[s1] + CF4A * cin[s2]);  cout += w;
            cin += s1;
        }
        cin += s1;
        *cout = CF(CF3A * cin[0] + CF3B * cin[-s1] + CF3C * cin[-s2]);  cout += w;
        *cout = CF(CF3X * cin[0] + CF3Y * cin[-s1] + CF3Z * cin[-s2]);  cout += w;
        *cout = CF(CF2A * cin[0] + CF2B * cin[-s1]);
    }
    c->height <<= 1;
    c->stride = c->width;
    delete [] c->pixels;
    c->pixels = out;
}

void JPEG::toRGB() {
    int i;
    Component *c;
    for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
        while(c->width < data.width)
            upSampleH(c);
        while(c->height < data.height)
            upSampleV(c);
    }
    if (data.comp_number == 3) {
        int x, yy;
        unsigned char *prgb = data.rgb;
        const unsigned char *py  = data.comp[0].pixels;
        const unsigned char *pcb = data.comp[1].pixels;
        const unsigned char *pcr = data.comp[2].pixels;
        for (yy = data.height; yy;  --yy) {
            for (x = 0;  x < data.width; ++x) {
                int y = py[x] << 8;
                int cb = pcb[x] - 128;
                int cr = pcr[x] - 128;
                *prgb++ = clip((y + 359 * cr + 128) >> 8);
                *prgb++ = clip((y - 88 * cb - 183 * cr + 128) >> 8);
                *prgb++ = clip((y + 454 * cb + 128) >> 8);
            }
            py += data.comp[0].stride;
            pcb += data.comp[1].stride;
            pcr += data.comp[2].stride;
        }
    } else if (data.comp[0].width != data.comp[0].stride) {
        unsigned char *pin = &data.comp[0].pixels[data.comp[0].stride];
        unsigned char *pout = &data.comp[0].pixels[data.comp[0].width];
        int y;
        for (y = data.comp[0].height - 1;  y;  --y) {
            memcpy(pout, pin, data.comp[0].width);
            pin += data.comp[0].stride;
            pout += data.comp[0].width;
        }
        data.comp[0].stride = data.comp[0].width;
    }
}

void JPEG::IDCTRow(int* mcu) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = mcu[4] << 11)
          | (x2 = mcu[6])
          | (x3 = mcu[2])
          | (x4 = mcu[1])
          | (x5 = mcu[7])
          | (x6 = mcu[5])
          | (x7 = mcu[3])))
    {
        mcu[0] = mcu[1] = mcu[2] = mcu[3] = mcu[4] = mcu[5] = mcu[6] = mcu[7] = mcu[0] << 3;
        return;
    }
    x0 = (mcu[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    mcu[0] = (x7 + x1) >> 8;
    mcu[1] = (x3 + x2) >> 8;
    mcu[2] = (x0 + x4) >> 8;
    mcu[3] = (x8 + x6) >> 8;
    mcu[4] = (x8 - x6) >> 8;
    mcu[5] = (x0 - x4) >> 8;
    mcu[6] = (x3 - x2) >> 8;
    mcu[7] = (x7 - x1) >> 8;
}

void JPEG::IDCTCol(const int* mcu, unsigned char *out, int stride) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = mcu[8 * 4] << 8)
          | (x2 = mcu[8 * 6])
          | (x3 = mcu[8 * 2])
          | (x4 = mcu[8 * 1])
          | (x5 = mcu[8 * 7])
          | (x6 = mcu[8 * 5])
          | (x7 = mcu[8 * 3])))
    {
        x1 = clip(((mcu[0] + 32) >> 6) + 128);
        for (x0 = 8; x0; --x0) {
            *out = (unsigned char) x1;
            out += stride;
        }
        return;
    }
    x0 = (mcu[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    *out = clip(((x7 + x1) >> 14) + 128);  out += stride;
    *out = clip(((x3 + x2) >> 14) + 128);  out += stride;
    *out = clip(((x0 + x4) >> 14) + 128);  out += stride;
    *out = clip(((x8 + x6) >> 14) + 128);  out += stride;
    *out = clip(((x8 - x6) >> 14) + 128);  out += stride;
    *out = clip(((x0 - x4) >> 14) + 128);  out += stride;
    *out = clip(((x3 - x2) >> 14) + 128);  out += stride;
    *out = clip(((x7 - x1) >> 14) + 128);
}

int JPEG::GetBits(int bits) {
    int res = showBits(bits);
    skipBits(bits);
    return res;
}

void JPEG::skipBits(int bits) {
    if (data.bufbits < bits)
        showBits(bits);
    data.bufbits -= bits;
}

int JPEG::showBits(int bits) {
    unsigned char neu;
    if (!bits)
        return 0;
    while(data.bufbits < bits) {
        neu = *(data.pos++);
        data.size--;
        if (neu == 0xFF) {
            unsigned char check = *data.pos++;
            if (check != 0x00 && check != 0xD9) {
                data.status = SYNTAX_ERROR;
            }
        }
        data.buf = (data.buf << 8) | neu;
        data.bufbits += 8;
    }
    return (data.buf >> (data.bufbits - bits)) & ((1 << bits) - 1);
}

int JPEG::GetVLC(VlcCode* vlc, unsigned char* code) {
    int value = showBits(16);
    int bits = vlc[value].bits;
    skipBits(bits);
    value = vlc[value].code;
    if (code)
        *code = (unsigned char)value;
    bits = value & 15;
    if (!bits)
        return 0;
    value = GetBits(bits);
    if (value < (1 << (bits - 1)))
        value += ((-1) << bits) + 1;
    return value;
}

void JPEG::decodeMCU(Component *c, unsigned char *out) {
    unsigned char code;
    int value, coef = 0;
    memset(data.mcu, 0, sizeof(data.mcu));
    c->dc += GetVLC(&data.vlctab[c->dctable][0], NULL);
    data.mcu[0] = (c->dc) * data.qtab[c->quant][0];
    do {
        value = GetVLC(&data.vlctab[c->actable][0], &code);
        if (!code)
            break;
        coef += (code >> 4) + 1;
        data.mcu[(int)ZigZag[coef]] = value * data.qtab[c->quant][coef];
    } while (coef < 63);
    for (coef = 0; coef < 64; coef += 8)
    IDCTRow(&data.mcu[coef]);
    for (coef = 0;  coef < 8;  ++coef)
    IDCTCol(&data.mcu[coef], &out[coef], c->stride);
}

void JPEG::readDATA() {
    int mcuh, mcuw, h, w, i;
    Component *c;
    for (mcuh = 0; mcuh < data.mcuheight; ++mcuh) {
        for (mcuw = 0; mcuw < data.mcuwidth; ++mcuw) {
            for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
                for (h = 0; h < c->sampley; ++h) {
                    for (w = 0; w < c->samplex; ++w) {
                        decodeMCU(c, &c->pixels[((mcuh * c->sampley + h) * c->stride + mcuw * c->samplex + w) << 3]);
                    }
                }
                
                // DRI
            }
        }
    }
    data.status = DECODE_FINISH;
}

void JPEG::readSOS() {
    int i;
    Component* c;
    GetLength();
    skip(1);
    for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
        c->dctable = data.pos[1] >> 4;
        c->actable = (data.pos[1] & 1) | 2;
        skip(2);
    }
    //    for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
    //        printf("id: %d dc: %d ac: %d\n", i, c->dctable, c->actable);
    //    }
    skip(data.length);
}

void JPEG::readDRI() {
    GetLength();
    data.resetinterval = Decode16(data.pos);
    skip(data.length);
}

void JPEG::readDHT() {
    int codelen, currcnt, remain, spread, i, j;
    VlcCode *vlc;
    unsigned char counts[16];
    GetLength();
    while (data.length >= 17) {
        i = data.pos[0];
        i = (i | (i >> 3)) & 3;  // combined DC/AC + tableid value
        for (codelen = 1; codelen <= 16; ++codelen)
        counts[codelen - 1] = data.pos[codelen];
        skip(17);
//        printf("DHT id: %d\n", i);
        vlc = &data.vlctab[i][0];
        remain = spread = 65536;
        for (codelen = 1; codelen <= 16; ++codelen) {
            spread >>= 1;
            currcnt = counts[codelen - 1];
            if (!currcnt) continue;
            remain -= currcnt << (16 - codelen);
            for (i = 0;  i < currcnt;  ++i) {
                unsigned char code = data.pos[i];
                for (j = spread; j; --j) {
                    vlc->bits = (unsigned char) codelen;
                    vlc->code = code;
                    ++vlc;
                }
            }
            skip(currcnt);
        }
        while (remain--) {
            vlc->bits = 0;
            ++vlc;
        }
    }
    if (data.length) {
        printf("read DHT Syntax Error!\n");
        data.status = SYNTAX_ERROR;
        return;
    }
}

void JPEG::readSOF0() {
    int i;
    Component *c;
    unsigned char precision;
    GetLength();
    precision = data.pos[0];
    data.height = Decode16(data.pos + 1);
    data.width = Decode16(data.pos + 3);
//    printf("resolution: %d * %d\n", data.width, data.height);
    data.comp_number = data.color_mode = data.pos[5];
    skip(6);
    for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
        c->id = data.pos[0];
        c->samplex = data.pos[1] >> 4;
        c->sampley = data.pos[1] & 0x0F;
        c->quant = data.pos[2];
//        printf("ID: %d width: %d height: %d quant: %d\n", c->id, c->samplex, c->sampley, c->quant);
        data.samplemaxx = (data.samplemaxx > c->samplex ? data.samplemaxx : c->samplex);
        data.samplemaxy = (data.samplemaxy > c->sampley ? data.samplemaxy : c->sampley);
        skip(3);
    }
    data.mcusizex = data.samplemaxx << 3;
    data.mcusizey = data.samplemaxy << 3;
    data.mcuwidth = (data.width + data.mcusizex - 1) / data.mcusizex;
    data.mcuheight = (data.height + data.mcusizey - 1) / data.mcusizey;
//    printf("mcuX: %d mcuY: %d mcuW: %d mcuH: %d\n", data.mcusizex, data.mcusizey, data.mcuwidth, data.mcuheight);
    for (i = 0, c = data.comp; i < data.comp_number; ++i, ++c) {
        c->width = (data.width * c->samplex + data.samplemaxx - 1) / data.samplemaxx;
        c->height = (data.height * c->sampley + data.samplemaxy - 1) / data.samplemaxy;
        c->stride = data.mcuwidth * data.mcusizex * c->samplex / data.samplemaxx;
        if (!(c->pixels = new unsigned char [(c->stride * (data.mcuheight * data.mcusizey * c->sampley / data.samplemaxy))])) {
            printf("Out of memory!\n");
            return;
        }
//        printf("id: %d width: %d height: %d\n", i, data.mcuwidth * data.mcusizex * c->samplex / data.samplemaxx, data.mcuheight * data.mcusizey * c->sampley / data.samplemaxy);
    }
    if (data.comp_number == 3) {
        data.rgb = new unsigned char [data.width * data.height * data.comp_number];
    }
    if (data.length) {
        printf("read SOF0 Syntax Error!\n");
        data.status = SYNTAX_ERROR;
        return;
    }
}

void JPEG::readDQT() {
    unsigned char c, precision, id;
    unsigned char *table;
    GetLength();
    while(data.length) {
        c = data.pos[0];
        precision = c >> 4 == 0 ? 8 : 16;
        //        printf("precision: %d\n", precision);
        precision /= 8;
        id = c & 0x0F;
//        printf("quan ID: %d\n", id);
        table = &data.qtab[id][0];
        skip(1);
        for (c = 0; c < 64; ++c) {
            table[c] = data.pos[c];
        }
        skip(64);
    }
    if (data.length) {
        printf("read DQT Syntax Error!\n");
        data.status = SYNTAX_ERROR;
        return;
    }
}

void JPEG::skipMARKER() {
    GetLength();
    skip(data.length);
}

void JPEG::readAPP1() {
    unsigned int IFD0_ptr;
    unsigned int Tag_num;
    
    GetLength();
    if (data.pos[0] == 'E' && data.pos[1] == 'x' && data.pos[6] == 'I' && data.pos[7] == 'I') {
        //        printf("Get Exif\n");
        skip(6);
        IFD0_ptr = Get32u(data.pos + 4);
        Tag_num = Get16u(data.pos + IFD0_ptr);
        if (Exif)
            skip(data.length - 6);
        Exif = new DATA_SET(Tag_num, data.pos);
        skip(IFD0_ptr + 2);
        readDataSet(Exif, data.pos);
    } else {
        //        printf("Unsupport APP1 format!\n");
    }
    skip(data.length);
}

void JPEG::showPicInfo() {
    if (Exif == NULL) {
        printf("No information!\n");
        return;
    }
    Exif->show();
}

void JPEG::readDataSet(DATA_SET *d, unsigned char *read_pos) {
    int sub_count = -1;
    for (int i = 0; i < d->Tag_num; ++i) {
        d->Tag[i] = Get16u(read_pos + (i * 12));
        d->Format[i] = Get16u(read_pos + (i * 12 + 2));
        d->Components[i] = Get32u(read_pos + (i * 12 + 4));
        d->Offset[i] = Get32u(read_pos + (i * 12 + 8));
        switch (d->Tag[i]) {
            case 34665:
            case 34853: ++sub_count;
            default: break;
        }
        d->Sub[i] = sub_count;
    }
    d->sub = new DATA_SET[sub_count + 1];
    for (int i = 0; i < d->Tag_num; ++i) {
        if (d->Sub[i] != -1) {
            unsigned int len = Get16u(d->start_pos + d->Offset[i]);
            d->sub[d->Sub[i]].Init(len, d->start_pos);
            switch (d->Tag[i]) {
                case 34665: d->sub[d->Sub[i]].method = DATA_SET::EXIF; break;
                case 34853: d->sub[d->Sub[i]].method = DATA_SET::GPS; break;
                default: break;
            }
            readDataSet(&d->sub[d->Sub[i]], d->start_pos + d->Offset[i] + 2);
        }
    }
}

JPEG::Status JPEG::decode() {
    //    printf("%d %d\n", data.pos[0], data.pos[1]);
    if ((data.pos[0] ^ 0xFF) | (data.pos[1] ^ 0xD8)) {
        printf("NOT JPEG!\n");
        return NOT_JPEG;
    }
    skip(2);
    while(!data.status) {
        skip(2);
        switch (data.pos[-1]) {
            case APP0_MARKER:
//                printf("APP0 MARKER\n");
                skipMARKER();
                break;
            case APP1_MARKER:
//                printf("APP1 MARKER\n");
                readAPP1();
                break;
            case DQT_MARKER:
//                printf("DQT MARKER\n");
                readDQT();
                break;
            case SOF0_MARKER:
//                printf("SOF0 MARKER\n");
                readSOF0();
                break;
            case SOF2_MARKER:
//                printf("SOF2 MARKER\n");
                break;
            case DHT_MARKER:
//                printf("DHT MARKER\n");
                readDHT();
                break;
            case SOS_MARKER:
//                printf("SOS MARKER\n");
                readSOS();
                readDATA();
                toRGB();
                break;
            case DRI_MARKER:
                //                printf("DRI MARKER\n");
                readDRI();
                break;
            default:
                if ((data.pos[-1] & 0xF0) == 0xE0)
                    skipMARKER();
                else
                    return UNSUPPORT;
                
        }
    }
    if (data.status == DECODE_FINISH)
        return OK;
    return data.status;
}

inline void JPEG::skip(int number) {
    data.pos += number;
    data.size -= number;
    data.length -= number;
}

void JPEG::GetLength() {
    data.length = Decode16(data.pos);
    skip(2);
}

inline unsigned short Decode16(const unsigned char *pos) {
    return (pos[0] << 8) | pos[1];
}

inline unsigned short Get16u(const unsigned char *pos) {
    return (pos[1] << 8) | pos[0];
}

inline unsigned int Get32u(const unsigned char *pos) {
    return (pos[3] << 24) | (pos[2] << 16) | (pos[1] << 8) | pos[0];
}

inline int Get32s(const unsigned char *pos) {
    return (pos[3] << 24) | (pos[2] << 16) | (pos[1] << 8) | pos[0];
}

unsigned char JPEG::clip(const int x) {
    return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}

void DATA_SET::str(unsigned char *pos, unsigned int len) {
    for (int i = 0; i < len; ++i) {
        printf("%c", pos[i]);
    }
    printf("\n");
}

float DATA_SET::ration64u(unsigned char *pos) {
    unsigned int up, down;
    up = Get32u(pos);
    down = Get32u(pos + 4);
    return 1.0 * up / down;
}

float DATA_SET::ration64s(unsigned char *pos) {
    int up, down;
    up = Get32s(pos);
    down = Get32s(pos + 4);
    return 1.0 * up / down;
}

void DATA_SET::show() {
    unsigned char *ptr;
    switch (method) {
        case TIFF: printf("*************一般資訊*************\n"); break;
        case EXIF: printf("*************EXIF*************\n"); break;
        case GPS: printf("*************GPS*************\n"); break;
        default: break;
    }
    
    for (int i = 0; i < Tag_num; ++i) {
        //        printf("Tag: %d Format: %d Components: %d Offset: %d\n", Tag[i], Format[i], Components[i], Offset[i]);
        ptr = start_pos + Offset[i];
        unsigned char c;
        switch (Tag[i]) {
            case 271:
                printf("製造商: "); break;
            case 272:
                printf("型號: "); break;
            case 274:
                printf("轉向: ");
                switch (Offset[i]) {
                    case 1: printf("水平\n"); break;
                    case 2: printf("水平鏡像\n"); break;
                    case 3: printf("旋轉180度\n"); break;
                    case 4: printf("垂直鏡像\n"); break;
                    case 5: printf("水平鏡像順時針旋轉270度\n"); break;
                    case 6: printf("順時針旋轉90度\n"); break;
                    case 7: printf("水平鏡像順時針旋轉90度\n"); break;
                    case 8: printf("順時針旋轉270度\n"); break;
                    default: break;
                }
                break;
            case 282:
                printf("x解析度: "); break;
            case 283:
                printf("y解析度: "); break;
            case 296:
                printf("解析度單位: ");
                switch (Offset[i]) {
                    case 1: printf("無\n"); break;
                    case 2: printf("英吋\n"); break;
                    case 3: printf("公分\n"); break;
                    default: break;
                }
                break;
            case 305: printf("軟體: "); break;
            case 306: printf("修改日期: "); break;
            case 315: printf("Artist: "); break;
            case 531:
                printf("YCbCrPositioning: ");
                switch (Offset[i]) {
                    case 1: printf("center of pixel array\n"); break;
                    case 2: printf("datum point\n"); break;
                    default: break;
                }
                break;
            case 33432: printf("Copyright: "); break;
            case 0: printf("GPS版本: "); break;
            case 1: printf("緯度: "); break;
            case 2: break;
            case 3: printf("經度: "); break;
            case 4: break;
            case 5: printf("海拔: "); break;
            case 6: break;
            case 7: printf("日期標記: "); break;
            case 8: printf("衛星: "); break;
            case 9: printf("狀態: "); break;
            case 10: printf("測量模式: "); break;
            case 11: printf("精準度: "); break;
            case 18: printf("地圖基準面: "); break;
            case 29: printf("日期標記: "); break;
            case ExposureTime: printf("曝光時間: "); break;
            case FNumber: printf("光圈: "); break;
            case ExposureProgram:
                printf("曝光模式: ");
                switch (Offset[i]) {
                    case 0: printf("未定義\n"); break;
                    case 1: printf("手動\n"); break;
                    case 2: printf("程式自動\n"); break;
                    case 3: printf("光圈優先\n"); break;
                    case 4: printf("快門優先\n"); break;
                    case 5: printf("Creative\n"); break;
                    case 6: printf("Action\n"); break;
                    case 7: printf("Protrait\n"); break;
                    case 8: printf("地景\n"); break;
                    case 9: printf("B快門\n"); break;
                    default: break;
                }
                break;
            case ISOSpeedRatings: printf("ISO: %u\n", Offset[i]); break;
            case SensitivityType:
                printf("SensitivityType: ");
                switch (Offset[i]) {
                    case 0: printf("未知\n"); break;
                    case 1: printf("Standard Output Sensitivity\n"); break;
                    case 2: printf("Recommended Exposure Index\n"); break;
                    case 3: printf("ISO Speed\n"); break;
                    case 4: printf("Standard Output Sensitivity and Recommended Exposure Index\n"); break;
                    case 5: printf("Standard Output Sensitivity and ISO Speed\n"); break;
                    case 6: printf("Recommended Exposure Index and ISO Speed\n"); break;
                    case 7: printf("Standard Output Sensitivity, Recommended Exposure Index and ISO Speed\n"); break;
                    default:
                        break;
                }
                break;
            case RecommendedExposureIndex: printf("RecommendedExposureIndex: %u\n", Offset[i]); break;
            case ExifVersion: break;
            case DateTimeOriginal: printf("DateTimeOriginal: "); break;
            case CreateDate: printf("創建日期: "); break;
            case OffsetTime: printf("修改日期時區: "); break;
            case OffsetTimeOriginal: printf("DateTimeOriginal日期時區: "); break;
            case OffsetTimeDigitized: printf("創建日期時區: "); break;
            case ShutterSpeedValue: printf("快門: "); break;
            case ApertureValue: printf("光圈值: "); break;
            case ExposureBiasValue: printf("曝光補償: "); break;
            case MaxApertureValue: printf("最大光圈值: "); break;
            case MeteringMode:
                printf("測光模式: ");
                switch (Offset[i]) {
                    case 0: printf("未知\n"); break;
                    case 1: printf("平均\n"); break;
                    case 2: printf("中央權衡\n"); break;
                    case 3: printf("Spot\n"); break;
                    case 4: printf("Multi-spot\n"); break;
                    case 5: printf("Multi-segment\n"); break;
                    case 6: printf("Partial\n"); break;
                    case 255: printf("其他\n"); break;
                    default:
                        break;
                }
                break;
            case Flash:
                printf("閃光燈: ");
                switch (Offset[i]) {
                    case 0x0: printf("無\n"); break;
                    case 0x1: printf("Fired\n"); break;
                    case 0x5: printf("Fired, Return not detected\n"); break;
                    case 0x7: printf("Fired, Return detected\n"); break;
                    case 0x8: printf("On, Did not fire\n"); break;
                    case 0x9: printf("On, Fired\n"); break;
                    case 0xd: printf("On, Return not detected\n"); break;
                    case 0xf: printf("On, Return detected\n"); break;
                    case 0x10: printf("Off, Did not fire\n"); break;
                    case 0x14: printf("Off, Did not fire, Return not detected\n"); break;
                    case 0x18: printf("Auto, Did not fire\n"); break;
                    case 0x19: printf("Auto, Fired\n"); break;
                    case 0x1d: printf("Auto, Fired, Return not detected\n"); break;
                    case 0x1f: printf("Auto, Fired, Return detected\n"); break;
                    case 0x20: printf("No flash function\n"); break;
                    case 0x30: printf("Off, No flash function\n"); break;
                    case 0x41: printf("Fired, Red-eye reduction\n"); break;
                    case 0x45: printf("Fired, Red-eye reduction, Return not detected\n"); break;
                    case 0x47: printf("Fired, Red-eye reduction, Return detected\n"); break;
                    case 0x49: printf("On, Red-eye reduction\n"); break;
                    case 0x4d: printf("On, Red-eye reduction, Return not detected\n"); break;
                    case 0x4f: printf("On, Red-eye reduction, Return detected\n"); break;
                    case 0x50: printf("Off, Red-eye reduction\n"); break;
                    case 0x58: printf("Auto, Did not fire, Red-eye reduction\n"); break;
                    case 0x59: printf("Auto, Fired, Red-eye reduction\n"); break;
                    case 0x5d: printf("Auto, Fired, Red-eye reduction, Return not detected\n"); break;
                    case 0x5f: printf("Auto, Fired, Red-eye reduction, Return detected\n"); break;
                    default:
                        break;
                }
                break;
            case FocalLength: printf("焦距: "); break;
            case SubSecTime: printf("SubSecTime: "); break;
            case SubSecTimeOriginal: printf("SubSecTimeOriginal: "); break;
            case SubSecTimeDigitized: printf("SubSecTimeDigitized: "); break;
            case FlashPixVersion: printf("FlashPixVersion: "); break;
            case ColorSpace:
                printf("色彩空間: ");
                switch (Offset[i]) {
                    case 1: printf("SRGB\n"); break;
                    case 65535: printf("未矯正\n"); break;
                    default: break;
                }
                break;
            case PixelXDimension: printf("X像素數: %u\n", Offset[i]); break;
            case PixelYDimension: printf("Y像素數: %u\n", Offset[i]); break;
            case FocalPlaneXResolution: printf("焦點平面X解析度: "); break;
            case FocalPlaneYResolution: printf("焦點平面Y解析度: "); break;
            case FocalPlaneResolutionUnit:
                printf("焦點平面解析度單位: ");
                switch (Offset[i]) {
                    case 1: printf("無\n"); break;
                    case 2: printf("英吋\n"); break;
                    case 3: printf("公分\n"); break;
                    default: break;
                }
                break;
            case CustomRendered:
                printf("自定義渲染: ");
                switch (Offset[i]) {
                    case 0: printf("一般\n"); break;
                    case 1: printf("自訂\n"); break;
                    default: break;
                }
                break;
            case ExposureMode:
                printf("曝光模式: ");
                switch (Offset[i]) {
                    case 0: printf("自動曝光\n"); break;
                    case 1: printf("手動曝光\n"); break;
                    case 2: printf("Auto bracket\n"); break;
                    default: break;
                }
                break;
            case WhiteBalance:
                printf("白平衡: ");
                switch (Offset[i]) {
                    case 0: printf("自動\n"); break;
                    case 1: printf("手動\n"); break;
                    default: break;
                }
                break;
            case SceneCaptureType:
                printf("SceneCaptureType: ");
                switch (Offset[i]) {
                    case 0: printf("標準\n"); break;
                    case 1: printf("地景\n"); break;
                    case 2: printf("自拍\n"); break;
                    case 3: printf("夜景\n"); break;
                    default: break;
                }
                break;
            case CameraOwnerName: printf("相機擁有者: "); break;
            case BodySerialNumber: printf("機身序號: "); break;
            case LensSpecification: printf("鏡頭焦距: "); break;
            case LensModel: printf("鏡頭型號: "); break;
            case LensSerialNumber: printf("鏡頭序號: "); break;
            default: break;
        }
        switch (method) {
            case TIFF:
                switch (Format[i]) {
                    case 5: printf("%g\n", ration64u(ptr)); break;
                    case 2: str(ptr, Components[i]); break;
                    default:
                        break;
                }
                break;
            case EXIF:
                switch (Format[i]) {
                    case 10:
                        printf("%g\n", pow(2, -(ration64s(ptr))));
                        break;
                    case 5: printf("%g\n", ration64u(ptr)); break;
                    case 2: str(ptr, Components[i]); break;
                    default:
                        break;
                }
                break;
            case GPS:
                switch (Format[i]) {
                    case 5:
                        for (int k = 0; k < Components[i]; ++k) {
                            printf("%g", ration64u(ptr + k * 8));
                            if (k != Components[i] - 1)
                                printf(", ");
                        }
                        printf("\n");
                        break;
                    case 2:
                        if (Components[i] <= 4) {
                            for (int k = 0; k < Components[i]; ++k) {
                                c = Offset[i] >> k * 8;
                                printf("%c", c);
                            }
                            printf("\n");
                        } else {
                            str(ptr, Components[i]);
                        }
                        break;
                    case 1:
                        for (int k = 0; k < Components[i]; ++k) {
                            c = Offset[i] >> k * 8;
                            printf("%d", c);
                            if (k != Components[i] - 1)
                                printf(".");
                        }
                        printf("\n");
                        break;
                    default: break;
                }
                break;
            default:
                break;
        }
    }
    for (int i = 0; i < Tag_num; ++i) {
        if (Sub[i] != -1) {
            sub[Sub[i]].show();
        }
    }
}


#endif /* jpeg_h */
