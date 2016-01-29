#include <math.h>
#include "mml.h"

//Mahalanobis' Distance
md::md(){
  int i,j;
  sample_num=0;
  for(i=0;i<3;i++){
    mv[i]=0;
    sample_sum[i]=0;
    for(j=0;j<3;j++){
      var[i][j]=0;
    }
  }
}
                        
void md::Sample_update(int X, int Y, int Z){

  sample_num++;
  sample_mean(X, Y, Z);
  sample_var(X, Y, Z);

}

void md::sample_mean(int X, int Y, int Z){

    sample_sum[0]+=X;
    sample_sum[1]+=Y;
    sample_sum[2]+=Z;

    mv[0]=sample_sum[0]/sample_num;
    mv[1]=sample_sum[1]/sample_num;
    mv[2]=sample_sum[2]/sample_num;
    
}

void md::sample_var(int X, int Y, int Z){
  float vXYZ[3];
  int i,j;

  vXYZ[0]=(float)(X-mv[0]);
  vXYZ[1]=(float)(Y-mv[1]);
  vXYZ[2]=(float)(Z-mv[2]);

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      covar_sum[i][j]+=vXYZ[i]*vXYZ[j];
      var[i][j]=covar_sum[i][j]/sample_num;
    }
  }

}

void md::rev_mtx(void){
 float det;
 det= var[0][0]*var[1][1]*var[2][2]+var[0][2]*var[2][1]*var[1][0]+var[2][0]*var[0][1]*var[1][2]-var[0][0]*var[2][1]*var[1][2]-var[2][0]*var[1][1]*var[0][2]-var[2][2]*var[0][1]*var[1][0];


 rvar[0][0]=(var[1][1]*var[2][2]-var[1][2]*var[2][1])/det;
 rvar[0][1]=(var[0][2]*var[2][1]-var[0][1]*var[2][2])/det;
 rvar[0][2]=(var[0][1]*var[1][2]-var[0][2]*var[1][1])/det;
 rvar[1][0]=(var[1][2]*var[2][0]-var[1][0]*var[2][2])/det;
 rvar[1][1]=(var[0][0]*var[2][2]-var[0][2]*var[2][0])/det;
 rvar[1][2]=(var[0][2]*var[1][0]-var[0][0]*var[1][2])/det;
 rvar[2][0]=(var[1][0]*var[2][1]-var[1][1]*var[2][0])/det;
 rvar[2][1]=(var[0][1]*var[2][0]-var[0][0]*var[2][1])/det;
 rvar[2][2]=(var[0][0]*var[1][1]-var[0][1]*var[1][0])/det;

}

float md::sample_mhlnbs( int X, int Y, int Z){
  float a1,a2,a3;
  float x1,x2,x3;
  
  x1=(float)(X-mv[0]);
  x2=(float)(Y-mv[1]);
  x3=(float)(Z-mv[2]);
  
  a1=x1*rvar[0][0]+x2*rvar[1][0]+x3*rvar[2][0];
  a2=x1*rvar[0][1]+x2*rvar[1][1]+x3*rvar[2][1];
  a3=x1*rvar[0][2]+x2*rvar[1][2]+x3*rvar[2][2];
  
  return x1*a1+x2*a2+x3*a3;
}



//Passive Aggressive
float dot(float a[], float b[], int length){
  float tmp=0;
  int i=0;

  for(i=0;i<length;i++) tmp+=a[i]*b[i];
  return tmp;
}

float hloss(int y, float w[], float x[], int length){
  float tmp;
  tmp=(float)y*dot(w,x,length);
  if(tmp<1) return 1-tmp;
  else return 0;
}

float PA(float C, float loss, float xn){
  return loss/xn;
}

float PA1(float C, float loss, float xn){
  return fmin(C, loss/xn);
}

float PA2(float C, float loss, float xn){
  return loss/(xn+1/(2*C));
}

// -----------------------------
// PA single member function
// -----------------------------
void pa_single::init(){
  int i;
  for(i=0;i<length;i++) w[i]=1/length;
}

pa_single::pa_single(const unsigned int size){
  w = new float[size];
  res = new float[size];
  length=size;
  init();
}

float* pa_single::get_w(){
  return w;
}

void pa_single::w_update(float x[], int y, float C){
  float yh;
  float xn;
  float tau;
  int i;
  yh=hloss(y,w,x,length);
  xn=dot(x,x,length);
  tau = PA1(C, yh, xn*xn);
  for(i=0;i<length;i++) w[i]+=y*tau*x[i];
}

int pa_single::pred(float x[]){
  if(dot(w,x,length)>0) return 1;
  else return -1;
}

