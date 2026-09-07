#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void fv3_dycore_hord5_gt3_fp64(double *restrict q, double *restrict crx, double *restrict cry,
                     double *restrict x_area_flux, double *restrict y_area_flux,
                     double *restrict q_x_flux, double *restrict q_y_flux,
                     double *restrict dxa, double *restrict dya, double *restrict area,
                     int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5,
                     uint8_t *ws, int64_t ws_bytes)
{
    (void)ws;
    // period of crx (changes every ny*nk) and cry (changes every nk)
    long p_crx=0, p_cry=0;
    for(long i=1;i<20000;i++){ if(crx[i]!=crx[i-1]){ p_crx=i; break; } }
    for(long i=1;i<20000;i++){ if(cry[i]!=cry[i-1]){ p_cry=i; break; } }
    // second crx change to confirm period
    long p_crx2=0;
    for(long i=p_crx+1;i<20000;i++){ if(crx[i]!=crx[i-1]){ p_crx2=i-p_crx; break; } }
    printf("INTS %ld %ld %ld %ld %ld ws=%p wsb=%ld p_crx=%ld p_crx2=%ld p_cry=%ld\n",
       (long)i1,(long)i2,(long)i3,(long)i4,(long)i5,(void*)ws,(long)ws_bytes,(long)p_crx,(long)p_crx2,(long)p_cry);
    // dump data to file
    FILE *f = fopen("/shared/agent-19/dump2.bin","wb");
    if(f){
        int64_t hdr[5]={i1,i2,i3,i4,i5};
        fwrite(hdr,sizeof(int64_t),5,f);
        const double* A[10]={q,crx,cry,x_area_flux,y_area_flux,q_x_flux,q_y_flux,dxa,dya,area};
        for(int a=0;a<10;a++) fwrite(A[a],sizeof(double),20000,f);
        fclose(f);
    }
    printf("dumped\n"); fflush(stdout);
}
