#include <cblas.h>
#include <stdio.h>
#include <stdint.h>

typedef struct
{
  float r;
  float i;
} complex_;

  typedef struct complex16 {
    float r;
    float i;
  } c16_t;

int main(void)
{
  c16_t A16[2][2] = {
                    {{0.7072,0.7072},{0,0}},
                    {{0,0},{0.7072,0.7072}}
  };
  c16_t B16[2][4] = {
                    {{1,0},{0.7072,0.7072},{0,1},{-0.7072,0.7072}},
                    {{2,0},{1.4142,1.4142},{0,2},{-1.4142,1.4142}}
  };
  c16_t C16[2][4];

  //複素数を構造体無しで扱う場合はaxpyを参照
  complex_ A[2][2];
  complex_ B[2][4]; // 4+3i 2+1i
  complex_ C[2][4]; //x * y =-6 +  48 i, -2 + 28 i
                    //        2 + 136 i,  6 + 84 i となることを確かめる

  //コピー c16_t -> complex_
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++)
      A[i][j] = (complex_){A16[i][j].r, A16[i][j].i};
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++) {
      B[i][j] = (complex_){B16[i][j].r, B16[i][j].i};
      C[i][j] = (complex_){0,0};
    }
  }
  
  complex_ alpha={1,0};
  complex_ beta={1,0};
  int i,j;
  cblas_cgemm(CblasRowMajor,//通常のCの行列であればCBlasRowMajorを指定
              CblasNoTrans, //A 転置しない場合 CblasNoTrans 転置 CblasTrans 共役転置 CblasConjTrans
              CblasNoTrans, //B 転置しない場合 CblasNoTrans 転置 CblasTrans 共役転置 CblasConjTrans
              2,            //Aの行数
              4,            //Bの列数
              2,            //Aの列数　＝　Bの行数（一致していないと掛け算できない）
              &alpha,       //alpha の値：構造体へのポインタになる
              A,            //A
              2,            //leading dimension (通常はAの列数)
              B,            //B
              4,            //leading dimension (通常はBの列数)
              &beta,        //beta の値：構造体へのポインタになる
              C,            //C
              4             //leading dimension (通常はCの列数)
              );

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 4; j++)
      C16[i][j] = (c16_t){C[i][j].r, C[i][j].i};
  }

  for(i=0;i<2;i++){
    for(j=0;j<4;j++){
        printf("%f + %f i, ",C16[i][j].r, C16[i][j].i);
    }
    printf("\n");
  }
  for(i=0;i<2;i++){
    for(j=0;j<4;j++){
        printf("%f + %f i, ",B[i][j].r, B[i][j].i);
    }
    printf("\n");
  }
  return 0;
}