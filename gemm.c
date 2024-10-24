#include <cblas.h>
#include <stdio.h>

int main(void)
{
  //double A[2*3]={1.0,2.0,3.0, 
  //               4.0,5.0,6.0};
  //double B[3*4]={1.0,2.0,3.0,4.0,
  //               5.0,6.0,7.0,8.0,
  //               9.0,0.0,1.0,2.0};
  //double C[2*4]; //x * y = 38 14 20 26
                 //        83 38 53 68 となることを確かめる
  
  //二次元配列の場合
  double A[2][3]={{1.0,2.0,3.0},
                  {4.0,5.0,6.0}}; //のように二次元配列で定義しても以下は同じコードで動く
  double B[3][4]={{1.0,2.0,3.0,4.0},
                  {5.0,6.0,7.0,8.0},
                  {9.0,0.0,1.0,2.0}};
  double C[2][4];
  
  int i,j;
  cblas_dgemm(CblasRowMajor,//通常のCの行列であればCBlasRowMajorを指定
              CblasNoTrans, //Aについて転置しない場合　CblasNoTrans 転置する場合　 CblasTrans
              CblasNoTrans, //Bについて転置しない場合　CblasNoTrans 転置する場合　 CblasTrans
              2,            //Aの行数
              4,            //Bの列数
              3,            //Aの列数　＝　Bの行数（一致していないと掛け算できない）
              1,            //alpha の値
              A,            //A
              3,            //leading dimension (通常はAの列数)
              B,            //B
              4,            //leading dimension (通常はBの列数)
              0,            //beta の値
              C,            //C
              4             //leading dimension (通常はCの列数)
              );

  for(i=0;i<2;i++){
    for(j=0;j<4;j++){
                                 //二次元配列の場合でもこの書き方でOK
        //printf("%lf ",C[i*4+j]); //(i,j)成分は i*ldC+jの形で書ける ldCはCのleading dimension :Cの列数
        printf("%lf ",C[i][j]);//もちろん、二次元配列で定義すればこれでも動く
    }
    printf("\n");
  }

  return 0;
}
