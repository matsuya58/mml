#ifndef __MML__
#define __MML__

//Mahalanobis' Distance
class md{
  private:
    int   mv[3];
    int   sample_sum[3];
    float var[3][3];
    float rvar[3][3];
    float covar_sum[3][3];
    int   sample_num;
    
  public:
    md(void);
    void sample_mean(int X, int Y, int Z);
    void sample_var(int X, int Y, int Z);
    void Sample_update(int X, int Y, int Z);
    void rev_mtx(void);
    float sample_mhlnbs( int X, int Y, int Z);
};


//Passive Aggressive
extern float dot(float a[], float b[], int length);
extern float hloss(float y, float w[], float x[], int length);

class pa_single{
  private:
    float *res;
    //float *w;
  protected:
    int length; 
    float *w;
  public:
    void init();
    pa_single(const unsigned int size);
    //~pa_single();
    void w_update(float x[], int y,float C);
    float* get_w();
    int pred(float x[]);
};


#endif
