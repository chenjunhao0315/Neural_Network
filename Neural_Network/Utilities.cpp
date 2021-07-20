//
//  Utilities.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/6/2.
//

#include "Utilities.hpp"

static bool return_v = false;
static float v_val = 0.0;

float randn(float mu, float std) {
    return mu + guassRandom() * std;
}

float guassRandom() {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    float u = Random();
    float v = Random();
    float r = u * u + v * v;
    if (r == 0 || r > 1)
        return guassRandom();
    float c = sqrt(-2 * log(r) / r);
    v_val = v * c;
    return_v = true;
    return u * c;
}

float Random() {
    return unif(generator);
}

void im2col_cpu(float* data_im, int channels, int height, int width, int ksize, int stride, int pad, float* data_col) {
    int c,h,w;
    /*得到輸出特徵圖的高和寬，其實這裏是不用算的，因爲在make_convolutional函數中已經算過，直接傳到這裏就好了*/
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    /*我們知道卷積運算時，我們是要用卷積對特徵圖所有通道都作卷積運算，因此這裏我們使用輸入通道數乘卷積核的大小，從而代表針對特徵圖同一位置卷積運算要用到的卷積核元素個數
    * 同時該變量也是轉換過後矩陣的行數
    */
    int channels_col = channels * ksize * ksize;
    /*以下三個循環決定了經過轉換的特徵圖矩陣的最終形式*/
    /*第一個循環表示轉換後矩陣的行數：輸入通道數*卷積核高*卷積核寬*/
    for (c = 0; c < channels_col; ++c) {
        /*以下三個偏移的計算就是要算出當前行的第一個元素在卷積核上對應的位置*/
        int w_offset = c % ksize; /*計算列偏移：卷積核是一個二維矩陣，並按行存儲在一維數組中，利用求餘運算獲取對應在卷積核中的列數*/
        int h_offset = (c / ksize) % ksize; /*計算行偏移*/
        int c_im = c / ksize / ksize;/*計算通道偏移*/
        /*接下來兩個循環就是個表示轉換後特徵矩陣的列數，即輸出特徵圖高*輸出特徵圖寬*/
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride; /*如果stride不爲1，那麼加上h*stride就是對對卷積核進行了移位操作*/
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;/*轉換後矩陣位置的索引*/
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int channels, int row, int col, int channel, int pad) {
    /*因爲當前給定的row和col是加了pad即補0之後的行列號，因此爲了得到真正的行列號，我們需要分別減去pad
    ** 注意，我們做pad時並非真的是在輸入特徵圖上補全0的行與列，而是在im2col轉化的過程中假裝輸入特徵圖裏面有0的行與列，之後在轉化後的結構中插入0
    */
    row -= pad;
    col -= pad;
    /*若出現判斷中的這四種情況，說明我們要取的數據在pad行或列中，最後輸出一定是0*/
    if (row < 0 || col < 0 || row >= height || col >= width)
        return 0;
    /*若要取得數據不在pad行或者pad列中，說明位於輸入特徵圖中，因此直接取出對應位置的數據就可以*/
    /*首先定位到對應的通道即width*height*channel,之後定位具體位置，即再加上col+width*row*/
    return im[col + width*(row + height * channel)];
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}
