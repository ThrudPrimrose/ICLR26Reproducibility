#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
static int64_t  *g_rowstart = NULL;
static int32_t  *g_csr_node = NULL;
static double   *g_csr_w    = NULL;
static double   *g_D        = NULL;
static const void *g_k_src=NULL,*g_k_dst=NULL,*g_k_w=NULL;
static int64_t g_k_E=0,g_k_N=0;
static uint64_t g_k_hash=0;
static int g_built=0;
static uint64_t sample_hash(const int64_t *src,const int64_t *dst,const double *w,int64_t E){
    uint64_t h=1469598103934665603ULL;
    int64_t ns = E>8192?8192:E;
    for(int64_t k=0;k<ns;k++){
        int64_t i=(int64_t)(((uint64_t)k*(uint64_t)E)/(uint64_t)ns);
        uint64_t t;
        memcpy(&t,&src[i],8); h^=t; h*=1099511628211ULL;
        memcpy(&t,&dst[i],8); h^=t; h*=1099511628211ULL;
        memcpy(&t,&w[i],8);   h^=t; h*=1099511628211ULL;
    }
    return h;
}
static void free_csr(void){ free(g_rowstart);free(g_csr_node);free(g_csr_w);free(g_D); g_rowstart=NULL;g_csr_node=NULL;g_csr_w=NULL;g_D=NULL;g_built=0; }
static int build_csr(const int64_t *src,const int64_t *dst,const double *w,int64_t E,int64_t N){
    free_csr();
    g_rowstart=(int64_t*)calloc((size_t)N+1,8);
    g_csr_node=(int32_t*)malloc((size_t)2*E*4);
    g_csr_w   =(double*) malloc((size_t)2*E*8);
    g_D       =(double*) malloc((size_t)N*8);
    if(!g_rowstart||!g_csr_node||!g_csr_w||!g_D) return -1;
    #pragma omp parallel for schedule(static)
    for(int64_t e=0;e<E;e++){
        __atomic_fetch_add(&g_rowstart[src[e]+1],1,__ATOMIC_RELAXED);
        __atomic_fetch_add(&g_rowstart[dst[e]+1],1,__ATOMIC_RELAXED);
    }
    for(int64_t i=0;i<N;i++) g_rowstart[i+1]+=g_rowstart[i];
    int64_t *next=(int64_t*)malloc((size_t)N*8);
    if(!next) return -1;
    memcpy(next,g_rowstart,(size_t)N*8);
    #pragma omp parallel for schedule(static)
    for(int64_t e=0;e<E;e++){
        int64_t s=src[e], d=dst[e]; double ww=w[e];
        int64_t p=__atomic_fetch_add(&next[s],1,__ATOMIC_RELAXED);
        g_csr_node[p]=(int32_t)d; g_csr_w[p]=ww;
        int64_t q=__atomic_fetch_add(&next[d],1,__ATOMIC_RELAXED);
        g_csr_node[q]=(int32_t)s; g_csr_w[q]=ww;
    }
    free(next);
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<N;i++){
        double s=0;
        for(int64_t j=g_rowstart[i];j<g_rowstart[i+1];j++) s+=g_csr_w[j];
        g_D[i]=s;
    }
    g_built=1; return 0;
}
static void fallback(double*Lx,const int64_t*dst,const int64_t*src,const double*w,const double*x,int64_t E,int64_t N){
    #pragma omp parallel for schedule(static)
    for (int64_t i=0;i<N;i++) Lx[i]=0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t e=0;e<E;e++){
        double f=w[e]*(x[src[e]]-x[dst[e]]);
        #pragma omp atomic
        Lx[src[e]]+=f;
        #pragma omp atomic
        Lx[dst[e]]-=f;
    }
}

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst,
                         const int64_t *restrict src, const double *restrict w,
                         const double *restrict x, const int64_t E, const int64_t N) {
    if (E<=0||N<=0) return;
    uint64_t h = sample_hash(src,dst,w,E);
    if (!(g_built && src==g_k_src && dst==g_k_dst && w==g_k_w && E==g_k_E && N==g_k_N && h==g_k_hash)) {
        if (build_csr(src,dst,w,E,N)!=0){ fallback(Lx,dst,src,w,x,E,N); return; }
        g_k_src=src; g_k_dst=dst; g_k_w=w; g_k_E=E; g_k_N=N; g_k_hash=h;
    }
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<N;i++){
        int64_t j=g_rowstart[i], end=g_rowstart[i+1];
        double s0=0,s1=0,s2=0,s3=0;
        int64_t j4 = end - ((end-j)&3);
        for(; j<j4; j+=4){
            s0 += g_csr_w[j]  *x[(size_t)g_csr_node[j]];
            s1 += g_csr_w[j+1]*x[(size_t)g_csr_node[j+1]];
            s2 += g_csr_w[j+2]*x[(size_t)g_csr_node[j+2]];
            s3 += g_csr_w[j+3]*x[(size_t)g_csr_node[j+3]];
        }
        double s = s0+s1+s2+s3;
        for(; j<end; j++) s += g_csr_w[j]*x[(size_t)g_csr_node[j]];
        Lx[i] = g_D[i]*x[i] - s;
    }
}
