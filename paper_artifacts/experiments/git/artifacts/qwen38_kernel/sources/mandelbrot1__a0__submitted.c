#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

void mandelbrot1_fp64(int64_t *restrict N_out, double _Complex *restrict Z_out,
                      int64_t maxiter, int64_t xn, int64_t yn,
                      uint8_t *restrict workspace, int64_t workspace_size) {
    (void)workspace; (void)workspace_size;
    const double xmin=-2.25, xmax=0.75, ymin=-1.25, ymax=1.25;
    const double horizon2 = 4.0;

    double *X = malloc((size_t)xn*sizeof(double));
    double *Y = malloc((size_t)yn*sizeof(double));
    double *TX = malloc((size_t)xn*sizeof(double));
    double *TY = malloc((size_t)yn*sizeof(double));
    if (xn>1){ double step=(xmax-xmin)/(double)(xn-1);
        for(int64_t i=0;i<xn;i++){ TX[i]=(double)i*step; }
        for(int64_t i=0;i<xn;i++){ X[i]=TX[i]+xmin; } X[xn-1]=xmax; }
    else X[0]=xmin;
    if (yn>1){ double step=(ymax-ymin)/(double)(yn-1);
        for(int64_t j=0;j<yn;j++){ TY[j]=(double)j*step; }
        for(int64_t j=0;j<yn;j++){ Y[j]=TY[j]+ymin; } Y[yn-1]=ymax; }
    else Y[0]=ymin;

    int64_t total = xn*yn;
    double *zr = malloc((size_t)total*sizeof(double));
    double *zi = malloc((size_t)total*sizeof(double));
    for(int64_t k=0;k<total;k++){ zr[k]=0.0; zi[k]=0.0; N_out[k]=0; }

    for(int64_t n=0;n<maxiter;n++){
        for(int64_t j=0;j<yn;j++){
            double yr=Y[j];
            for(int64_t i=0;i<xn;i++){
                int64_t idx=j*xn+i;
                double a=zr[idx], b=zi[idx];
                double a2=a*a, b2=b*b;
                double mag2=a2+b2;
                if(mag2<horizon2){
                    N_out[idx]=n;
                    double sr=fma(a,a,-(b*b));
                    double t=a*b;
                    double si=t+t;
                    zr[idx]=sr+X[i];
                    zi[idx]=si+yr;
                }
            }
        }
    }
    for(int64_t k=0;k<total;k++){
        if(N_out[k]==maxiter-1) N_out[k]=0;
        Z_out[k]=zr[k]+zi[k]*I;
    }
    free(X);free(Y);free(TX);free(TY);free(zr);free(zi);
}
