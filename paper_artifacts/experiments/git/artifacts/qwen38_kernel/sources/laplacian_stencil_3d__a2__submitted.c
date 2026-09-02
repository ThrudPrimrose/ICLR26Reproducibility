#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#define C03 (-8.5416666666666664)  /* 3 * (-205.0/72.0) */

void laplacian_stencil_3d_fp64(
    double *restrict ekin, double *restrict lap, const double *restrict psi,
    const int64_t N, const double inv_h2, const int64_t k,
    uint8_t *restrict workspace, const int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    const double w1 = 8.0/5.0, w2 = -1.0/5.0, w3 = 8.0/315.0, w4 = -1.0/560.0;
    const int64_t Nk = N*k, N2k = N*N*k;

    #pragma omp parallel
    {
        double *restrict ek = (double*)aligned_alloc(64, ((k+7)&~7)*sizeof(double));
        for (int64_t m=0;m<k;m++) ek[m]=0.0;
        #pragma omp for schedule(static)
        for (int64_t i=0;i<N;i++){
            const int64_t xd1=((i+1)%N-i)*N2k, xd2=((i+2)%N-i)*N2k, xd3=((i+3)%N-i)*N2k, xd4=((i+4)%N-i)*N2k;
            const int64_t xm1=((i-1+N)%N-i)*N2k, xm2=((i-2+N)%N-i)*N2k, xm3=((i-3+N)%N-i)*N2k, xm4=((i-4+N)%N-i)*N2k;
            for (int64_t j=0;j<N;j++){
                const int64_t yd1=((j+1)%N-j)*Nk, yd2=((j+2)%N-j)*Nk, yd3=((j+3)%N-j)*Nk, yd4=((j+4)%N-j)*Nk;
                const int64_t ym1=((j-1+N)%N-j)*Nk, ym2=((j-2+N)%N-j)*Nk, ym3=((j-3+N)%N-j)*Nk, ym4=((j-4+N)%N-j)*Nk;
                const double *base = psi + (i*N+j)*Nk;
                double *obase = lap + (i*N+j)*Nk;
                const double *X1=base+xd1,*Xm1=base+xm1,*X2=base+xd2,*Xm2=base+xm2,*X3=base+xd3,*Xm3=base+xm3,*X4=base+xd4,*Xm4=base+xm4;
                const double *Y1=base+yd1,*Ym1=base+ym1,*Y2=base+yd2,*Ym2=base+ym2,*Y3=base+yd3,*Ym3=base+ym3,*Y4=base+yd4,*Ym4=base+ym4;
                /* Pass 1a: interior l in [4,N-4) -- flat (l,m) full 8-wide, no wrap.
                   f indexes (l*k+m); z-neighbors are f±s*k (in-bounds here). */
                {
                    const int64_t fstart = 4*k, fend = (N-4)*k;
                    #pragma omp simd
                    for (int64_t f=fstart; f<fend; f++){
                        double v = C03*base[f];
                        v += w1*(base[f+k]+base[f-k]);
                        v += w2*(base[f+2*k]+base[f-2*k]);
                        v += w3*(base[f+3*k]+base[f-3*k]);
                        v += w4*(base[f+4*k]+base[f-4*k]);
                        v += w1*(Y1[f]+Ym1[f]);
                        v += w2*(Y2[f]+Ym2[f]);
                        v += w3*(Y3[f]+Ym3[f]);
                        v += w4*(Y4[f]+Ym4[f]);
                        v += w1*(X1[f]+Xm1[f]);
                        v += w2*(X2[f]+Xm2[f]);
                        v += w3*(X3[f]+Xm3[f]);
                        v += w4*(X4[f]+Xm4[f]);
                        obase[f] = inv_h2*v;
                    }
                }
                /* Pass 1b: boundary l in [0,4) and [N-4,N) -- wrap-aware, m-vectorized. */
                for (int64_t l=0;l<N;l++){
                    if (l>=4 && l<N-4) continue;
                    const int64_t zd1=((l+1)%N-l)*k, zd2=((l+2)%N-l)*k, zd3=((l+3)%N-l)*k, zd4=((l+4)%N-l)*k;
                    const int64_t zm1=((l-1+N)%N-l)*k, zm2=((l-2+N)%N-l)*k, zm3=((l-3+N)%N-l)*k, zm4=((l-4+N)%N-l)*k;
                    const double *c = base + l*k; double *o = obase + l*k;
                    const double *a1=c+xd1,*b1=c+xm1,*a2=c+xd2,*b2=c+xm2,*a3=c+xd3,*b3=c+xm3,*a4=c+xd4,*b4=c+xm4;
                    const double *p1=c+yd1,*q1=c+ym1,*p2=c+yd2,*q2=c+ym2,*p3=c+yd3,*q3=c+ym3,*p4=c+yd4,*q4=c+ym4;
                    const double *r1=c+zd1,*s1=c+zm1,*r2=c+zd2,*s2=c+zm2,*r3=c+zd3,*s3=c+zm3,*r4=c+zd4,*s4=c+zm4;
                    #pragma omp simd
                    for (int64_t m=0;m<k;m++){
                        double v = C03*c[m];
                        v += w1*(a1[m]+b1[m]); v += w2*(a2[m]+b2[m]); v += w3*(a3[m]+b3[m]); v += w4*(a4[m]+b4[m]);
                        v += w1*(p1[m]+q1[m]); v += w2*(p2[m]+q2[m]); v += w3*(p3[m]+q3[m]); v += w4*(p4[m]+q4[m]);
                        v += w1*(r1[m]+s1[m]); v += w2*(r2[m]+s2[m]); v += w3*(r3[m]+s3[m]); v += w4*(r4[m]+s4[m]);
                        o[m] = inv_h2*v;
                    }
                }
                /* Pass 2: ek = -0.5*sum psi*lap per m (cheap: 2 loads/element). */
                for (int64_t l=0;l<N;l++){
                    const double *c = base + l*k; double *o = obase + l*k;
                    #pragma omp simd
                    for (int64_t m=0;m<k;m++) ek[m] += c[m]*o[m];
                }
            }
        }
        for (int64_t m=0;m<k;m++){
            #pragma omp atomic
            ekin[m] += -0.5*ek[m];
        }
        free(ek);
    }
}
