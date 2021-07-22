//
//  Jpeg.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/7/15.
//

#include "Jpeg.hpp"

JPEG::~JPEG() {
    delete Exif;
    delete [] data.data_indicator;
    delete [] data.comp[0].pixels;
    delete [] data.comp[1].pixels;
    delete [] data.comp[2].pixels;
    if (data.status != DECODED_MODE)
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
    data.rgb = nullptr;
    Exif = nullptr;
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
//    switch (data.status) {
//        case OK: printf("Decode finish!\n"); break;
//        case NOT_JPEG: printf("Not jpeg file!\n"); break;
//        case SYNTAX_ERROR: printf("Syntax error!\n"); break;
//        case UNSUPPORT: printf("Unsupport!\n"); break;
//        default: break;
//    }
}

JPEG::JPEG(unsigned char *rgb, int width, int height, int channel) {
    data.comp[0].pixels = nullptr;
    data.comp[1].pixels = nullptr;
    data.comp[2].pixels = nullptr;
    data.rgb = rgb;
    Exif = nullptr;
    data.pos = nullptr;
    data.data_indicator = nullptr;
    data.width = width;
    data.height = height;
    data.comp_number = channel;
    char table[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
        11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
        42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
        38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
    memcpy(ZigZag, table, sizeof(ZigZag));
    data.status = DECODED_MODE;
}

void JPEG::convert_ppm(const char *filename) {
    if ((status() != OK) && (status() != DECODED_MODE))
        return;
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P%d\n%d %d\n255\n", data.comp_number == 3 ? 6 : 5, data.width, data.height);
    fwrite(data.rgb, 1, data.width * data.height * data.comp_number, f);
    fclose(f);
}

int JPEG::getWidth() {
    return data.width;
}

int JPEG::getHeight() {
    return data.height;
}

int JPEG::getChannel() {
    return data.comp_number;
}

unsigned char * JPEG::getPixel() {
    return data.rgb;
}

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
        if (i & 0xEC) {
            printf("DHT Syntax Error!\n");
            data.status = SYNTAX_ERROR;
            return;
        }
        if (i & 0x02) {
            printf("DHT unsupport!\n");
            data.status = UNSUPPORT;
            return;
        }
        i = (i | (i >> 3)) & 3;
        for (codelen = 1; codelen <= 16; ++codelen) {
            counts[codelen - 1] = data.pos[codelen];
        }
        skip(17);
        //        printf("DHT id: %d\n", i);
        vlc = &data.vlctab[i][0];
        remain = spread = 65536;
        for (codelen = 1; codelen <= 16; ++codelen) {
            spread >>= 1;
            currcnt = counts[codelen - 1];
            if (!currcnt) continue;
            remain -= currcnt << (16 - codelen);
            for (i = 0; i < currcnt; ++i) {
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
        printf("DHT Syntax Error!\n");
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
        if (c->samplex & (c->samplex - 1)) {
            printf("SOF0 samplex is non power of two!\n");
            data.status = UNSUPPORT;
            return;
        }
        if (c->sampley & (c->sampley - 1)) {
            printf("SOF0 sampley is non power of two!\n");
            data.status = UNSUPPORT;
            return;
        }
        if (!(c->samplex) || !(c->sampley)) {
            printf("SOF0 syntax error!\n");
            data.status = SYNTAX_ERROR;
            return;
        }
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
                data.status = UNSUPPORT;
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

template <typename Number, typename Limit>
Number clamp(Number value, Limit minValue, Limit maxValue) {
    if (value <= minValue)
        return minValue;
    if (value >= maxValue)
        return maxValue;
    return value;
}

void JPEG::generateHuffmanTable(const unsigned char numCodes[16], const unsigned char *values, BitCode result[256]) {
    auto huffmanCode = 0;
    for (int numBits = 1; numBits <= 16; numBits++) {
        for (int i = 0; i < numCodes[numBits - 1]; i++)
        result[*values++] = BitCode(huffmanCode++, numBits);
        huffmanCode <<= 1;
    }
}

float JPEG::convertRGBtoY(float r, float g, float b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

float JPEG::convertRGBtoCb(float r, float g, float b) {
    return -0.16874 * r - 0.33126 * g + 0.5 * b;
}

float JPEG::convertRGBtoCr(float r, float g, float b) {
    return 0.5 * r -0.41869 * g - 0.08131 * b;
}

Writer::Writer(const char *filename) {
    f = fopen(filename, "wb");
}

Writer::~Writer() {
    fclose(f);
}

void Writer::write(const void *data, int size) {
    fwrite(data, size, 1, f);
}

void Writer::write_byte(unsigned char data) {
    this->write(&data, 1);
}

void Writer::write_word(unsigned short data) {
    unsigned short data_ = ((data >> 8) & 0xFF) | ((data & 0xFF) << 8);
    this->write(&data_, 2);
}

void Writer::write_bits(const BitCode &data) {
    buffer.numBits += data.numBits;
    buffer.data   <<= data.numBits;
    buffer.data    |= data.code;
    while (buffer.numBits >= 8) {
        buffer.numBits -= 8;
        unsigned char oneByte = (unsigned char)(buffer.data >> buffer.numBits);
        write_byte(oneByte);
        if (oneByte == 0xFF)
            write_byte(0x00);
    }
}

void Writer::addMarker(unsigned char id, unsigned short length) {
    this->write_byte(0xFF);
    this->write_byte(id);
    this->write_word(length);
}

void Writer::flush() {
    this->write_bits(BitCode(0x7F, 7));
}

bool JPEG::save(const char *filename, float quality_, bool down_sample) {
    Writer writer(filename);
    if (data.rgb == nullptr)
        return false;
    if (data.width == 0 || data.height == 0)
        return false;
    
    // Header
    writer.write_word(0xFFD8);  // SOI
    writer.write_word(0xFFE0);  // JFIF APP0
    writer.write_word(16);
    writer.write("JFIF", 5);    // JFIF
    writer.write_byte(1);       // Version high bit 1
    writer.write_byte(1);       // Version low bit 1
    writer.write_byte(0);
    writer.write_word(1);
    writer.write_word(1);
    writer.write_byte(0);       // ThumbWidth
    writer.write_byte(0);       // ThumbHeight
    
    // DQT
    float quality = clamp(quality_, 1, 100);
    quality = quality < 50 ? 5000 / quality : 200 - quality * 2;
    unsigned char quantLuminance  [8*8];
    unsigned char quantChrominance[8*8];
    for (int i = 0; i < 64; ++i) {
        int luminance = (DefaultQuantLuminance[ZigZag[i]] * quality + 50) / 100;
        int chrominance = (DefaultQuantChrominance[ZigZag[i]] * quality + 50) / 100;
        
        quantLuminance[i] = clamp(luminance, 1, 255);
        quantChrominance[i] = clamp(chrominance, 1, 255);
    }
    writer.addMarker(DQT_MARKER, 132);
    writer.write_byte(0x00);
    writer.write(quantLuminance, 64);
    writer.write_byte(0x01);
    writer.write(quantChrominance, 64);
    
    // SOF0
    writer.addMarker(SOF0_MARKER, 2 + 6 + 3 * data.comp_number);
    writer.write_byte(0x08);
    writer.write_byte((unsigned char)(data.height >> 8));
    writer.write_byte((unsigned char)(data.height & 0xFF));
    writer.write_byte((unsigned char)(data.width >> 8));
    writer.write_byte((unsigned char)(data.width & 0xFF));

    writer.write_byte((unsigned char)(data.comp_number));
    for (unsigned char id = 1; id <= data.comp_number; ++id) {
        writer.write_byte(id);
        writer.write_byte((id == 1 && down_sample) ? 0x22 : 0x11);
        writer.write_byte((id == 1) ? 0x00 : 0x01);
    }
    
    // DHT
    writer.addMarker(DHT_MARKER, 418);
    writer.write_byte(0x00);
    writer.write(DcLuminanceCodesPerBitsize, sizeof(DcLuminanceCodesPerBitsize));
    writer.write(DcLuminanceValues, sizeof(DcLuminanceValues));
    writer.write_byte(0x10);
    writer.write(AcLuminanceCodesPerBitsize, sizeof(AcLuminanceCodesPerBitsize));
    writer.write(AcLuminanceValues, sizeof(AcLuminanceValues));
    writer.write_byte(0x01);
    writer.write(DcChrominanceCodesPerBitsize, sizeof(DcChrominanceCodesPerBitsize));
    writer.write(DcChrominanceValues, sizeof(DcChrominanceValues));
    writer.write_byte(0x11);
    writer.write(AcChrominanceCodesPerBitsize, sizeof(AcChrominanceCodesPerBitsize));
    writer.write(AcChrominanceValues, sizeof(AcChrominanceValues));
    
    BitCode huffmanLuminanceDC[256];
    BitCode huffmanLuminanceAC[256];
    generateHuffmanTable(DcLuminanceCodesPerBitsize, DcLuminanceValues, huffmanLuminanceDC);
    generateHuffmanTable(AcLuminanceCodesPerBitsize, AcLuminanceValues, huffmanLuminanceAC);
    BitCode huffmanChrominanceDC[256];
    BitCode huffmanChrominanceAC[256];
    generateHuffmanTable(DcChrominanceCodesPerBitsize, DcChrominanceValues, huffmanChrominanceDC);
    generateHuffmanTable(AcChrominanceCodesPerBitsize, AcChrominanceValues, huffmanChrominanceAC);
    
    // SOS
    writer.addMarker(SOS_MARKER, 2 + 1 + 2 * data.comp_number + 3);
    writer.write_byte(data.comp_number);
    for (unsigned char id = 1; id <= data.comp_number; ++id) {
        writer.write_byte(id);
        writer.write_byte((id == 1) ? 0x00 : 0x11);
    }
    writer.write_byte(0);
    writer.write_byte(63);
    writer.write_byte(0);
    
    // Bits stream
    float scaledLuminance[64];
    float scaledChrominance[64];
    for (int i = 0; i < 64; i++)
    {
        int row = ZigZag[i] / 8;
        int column = ZigZag[i] % 8;
        
        static const float AanScaleFactors[8] = {
            1, 1.387039845f, 1.306562965f, 1.175875602f, 1, 0.785694958f, 0.541196100f, 0.275899379f
        };
        float factor = 1 / (AanScaleFactors[row] * AanScaleFactors[column] * 8);
        scaledLuminance[ZigZag[i]] = factor / quantLuminance  [i];
        scaledChrominance[ZigZag[i]] = factor / quantChrominance[i];
    }
    
    BitCode  codewordsArray[2 * CodeWordLimit];
    BitCode* codewords = &codewordsArray[CodeWordLimit];
    unsigned char numBits = 1;
    int mask = 1;
    for (int value = 1; value < CodeWordLimit; value++) {
        if (value > mask) {
            numBits++;
            mask = (mask << 1) | 1;
        }
        codewords[-value] = BitCode(mask - value, numBits);
        codewords[+value] = BitCode(value, numBits);
    }
    
    unsigned char *pixels = data.rgb;
    const int width = data.width;
    const int height = data.height;
    const int maxWidth = width - 1;
    const int maxHeight = height - 1;
    
    const int sampling = (down_sample) ? 2 : 1;
    const int mcuSize = 8 * sampling;
    
    int lastYDC = 0, lastCbDC = 0, lastCrDC = 0;
    float Y[8][8], Cb[8][8], Cr[8][8];
    
    for (int mcuY = 0; mcuY < height; mcuY += mcuSize) {
        for (int mcuX = 0; mcuX < width; mcuX += mcuSize) {
            for (int blockY = 0; blockY < mcuSize; blockY += 8) {
                for (int blockX = 0; blockX < mcuSize; blockX += 8) {
                    
                    // Convert color space
                    for (int deltaY = 0; deltaY < 8; deltaY++)
                    {
                        int column = min(mcuX + blockX , maxWidth);
                        int row = min(mcuY + blockY + deltaY, maxHeight);
                        for (int deltaX = 0; deltaX < 8; deltaX++) {
                            int pixelPos = row * int(width) + column;
                            if (column < maxWidth)
                                column++;
                            unsigned char r = pixels[3 * pixelPos + 0];
                            unsigned char g = pixels[3 * pixelPos + 1];
                            unsigned char b = pixels[3 * pixelPos + 2];
                            
                            Y   [deltaY][deltaX] = convertRGBtoY(r, g, b) - 128;
                            if (!down_sample) {
                                Cb[deltaY][deltaX] = convertRGBtoCb(r, g, b);
                                Cr[deltaY][deltaX] = convertRGBtoCr(r, g, b);
                            }
                        }
                    }
                    
                    // encode Y channel
                    lastYDC = encodeBlock(writer, Y, scaledLuminance, lastYDC, huffmanLuminanceDC, huffmanLuminanceAC, codewords);
                    // Cb and Cr are encoded about 50 lines below
                }
                if (down_sample)
                    for (short deltaY = 7; down_sample && deltaY >= 0; deltaY--) {
                        int row = min(mcuY + 2 * deltaY, maxHeight);
                        int column = mcuX;
                        int pixelPos = (row * int(width) + column) * 3;
                        int rowStep = (row < maxHeight) ? 3 * int(width) : 0;
                        int columnStep = (column < maxWidth ) ? 3 : 0;
                        
                        for (short deltaX = 0; deltaX < 8; deltaX++) {
                            int right = pixelPos + columnStep;
                            int down = pixelPos + rowStep;
                            int downRight = pixelPos + columnStep + rowStep;
                            
                            unsigned char r = short(pixels[pixelPos]) + pixels[right] + pixels[down] + pixels[downRight];
                            unsigned char g = short(pixels[pixelPos + 1]) + pixels[right + 1] + pixels[down + 1] + pixels[downRight + 1];
                            unsigned char b = short(pixels[pixelPos + 2]) + pixels[right + 2] + pixels[down + 2] + pixels[downRight + 2];
                            
                            Cb[deltaY][deltaX] = convertRGBtoCb(r, g, b) / 4;
                            Cr[deltaY][deltaX] = convertRGBtoCr(r, g, b) / 4;
                            
                            pixelPos += 2 * 3;
                            column += 2;
                            
                            if (column >= maxWidth) {
                                columnStep = 0;
                                pixelPos = ((row + 1) * int(width) - 1) * 3;
                            }
                        }
                    }
                lastCbDC = encodeBlock(writer, Cb, scaledChrominance, lastCbDC, huffmanChrominanceDC, huffmanChrominanceAC, codewords);
                lastCrDC = encodeBlock(writer, Cr, scaledChrominance, lastCrDC, huffmanChrominanceDC, huffmanChrominanceAC, codewords);
            }
        }
    }
    
    writer.flush();

    // EOI
    writer.write_byte(0xFF);
    writer.write_byte(0xD9);
    return true;
}

int JPEG::encodeBlock(Writer& writer, float block[8][8], const float scaled[64], int lastDC, const BitCode huffmanDC[256], const BitCode huffmanAC[256], const BitCode* codewords) {
    float *block64 = (float*)block;
    
    for (int offset = 0; offset < 8; offset++) {
        DCT(block64 + offset * 8, 1);
    }
    for (int offset = 0; offset < 8; offset++) {
        DCT(block64 + offset * 1, 8);
    }
    
    for (int i = 0; i < 64; i++) {
        block64[i] *= scaled[i];
    }
    
    int dc = int(block64[0] + (block64[0] >= 0 ? +0.5 : -0.5));
    
    int posNonZero = 0;
    short int quantized[64];
    for (short int i = 1; i < 64; i++) {
        float value = block64[ZigZag[i]];
        quantized[i] = int(value + (value >= 0 ? +0.5f : -0.5f));
        if (quantized[i] != 0)
            posNonZero = i;
    }
    
    int diff = dc - lastDC;
    if (diff == 0)
        writer.write_bits(huffmanDC[0x00]);
    else {
        BitCode bits = codewords[diff];
        writer.write_bits(huffmanDC[bits.numBits]);
        writer.write_bits(bits);
    }
    
    int offset = 0;
    for (int i = 1; i <= posNonZero; i++) {
        while (quantized[i] == 0) {
            offset += 0x10;
            if (offset > 0xF0) {
                writer.write_bits(huffmanAC[0xF0]);
                offset = 0;
            }
            i++;
        }
        BitCode encoded = codewords[quantized[i]];
        writer.write_bits(huffmanAC[offset + encoded.numBits]);
        writer.write_bits(encoded);
        offset = 0;
    }
    
    if (posNonZero < 64 - 1)
        writer.write_bits(huffmanAC[0x00]);
    return dc;
}

void JPEG::DCT(float block[8*8], unsigned short stride) {
    const float SqrtHalfSqrt = 1.306562965;
    const float InvSqrt = 0.707106781;
    const float HalfSqrtSqrt = 0.382683432;
    const float InvSqrtSqrt = 0.541196100;
    
    float& block0 = block[0         ];
    float& block1 = block[1 * stride];
    float& block2 = block[2 * stride];
    float& block3 = block[3 * stride];
    float& block4 = block[4 * stride];
    float& block5 = block[5 * stride];
    float& block6 = block[6 * stride];
    float& block7 = block[7 * stride];
    
    float add07 = block0 + block7; float sub07 = block0 - block7;
    float add16 = block1 + block6; float sub16 = block1 - block6;
    float add25 = block2 + block5; float sub25 = block2 - block5;
    float add34 = block3 + block4; float sub34 = block3 - block4;
    
    float add0347 = add07 + add34; float sub07_34 = add07 - add34;
    float add1256 = add16 + add25; float sub16_25 = add16 - add25;
    
    block0 = add0347 + add1256; block4 = add0347 - add1256;
    
    float z1 = (sub16_25 + sub07_34) * InvSqrt;
    block2 = sub07_34 + z1; block6 = sub07_34 - z1;
    
    float sub23_45 = sub25 + sub34;
    float sub12_56 = sub16 + sub25;
    float sub01_67 = sub16 + sub07;
    
    float z5 = (sub23_45 - sub01_67) * HalfSqrtSqrt;
    float z2 = sub23_45 * InvSqrtSqrt  + z5;
    float z3 = sub12_56 * InvSqrt;
    float z4 = sub01_67 * SqrtHalfSqrt + z5;
    float z6 = sub07 + z3;
    float z7 = sub07 - z3;
    block1 = z6 + z4; block7 = z6 - z4;
    block5 = z7 + z2; block3 = z7 - z2;
}
