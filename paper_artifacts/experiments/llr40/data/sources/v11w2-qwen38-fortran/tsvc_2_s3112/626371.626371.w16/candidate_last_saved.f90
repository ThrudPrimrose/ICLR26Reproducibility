subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C, name='tsvc_2_s3112_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(kind=c_double), dimension(*), intent(in)  :: a
  real(kind=c_double), dimension(*), intent(out) :: b
  integer(kind=c_int64_t), value, intent(in)     :: len_1d

  interface
    integer function c_num_threads() bind(C, name='omp_get_num_threads')
    end function c_num_threads
    integer function c_thread_num() bind(C, name='omp_get_thread_num')
    end function c_thread_num
  end interface

  integer(kind=c_int64_t) :: n, nc, csize, k, i, i0, i1, ng, m, j
  real(kind=c_double)     :: cs(0:8191)
  real(kind=c_double)     :: s1,s2,s3,s4,s5,s6,s7,s8, t, v
  integer                 :: nt, tid
  integer(kind=c_int64_t), parameter :: MIN_CHUNK = 65536_8
  integer(kind=c_int64_t), parameter :: CHUNK_MULT = 16_8
  integer(kind=c_int64_t), parameter :: MIN_NCHUNK = 64_8

  n = len_1d
  if (n <= 0) return
  nt = c_num_threads()
  if (n < 100000_8) then
     nc = min(n, 2_8*nt)
  else
     nc = min(n, max(CHUNK_MULT*nt, MIN_NCHUNK))
     if (n / nc < MIN_CHUNK) nc = max(1_8, n / MIN_CHUNK)
  end if
  csize = (n + nc - 1_8) / nc
  cs(0) = 0.0d0

!$omp parallel shared(a,b,n,nc,csize,cs) private(nt,tid,k,i,i0,i1,ng,m,j,s1,s2,s3,s4,s5,s6,s7,s8,t,v)
  nt = c_num_threads()
  tid = c_thread_num()
  do k = tid, nc-1, nt
     i0 = k*csize + 1
     m  = min(csize, n - k*csize + 1)
     ng = m / 8
     s1 = 0.0d0; s2 = 0.0d0; s3 = 0.0d0; s4 = 0.0d0
     s5 = 0.0d0; s6 = 0.0d0; s7 = 0.0d0; s8 = 0.0d0
     do i = i0, i0 + 8*ng - 8, 8
        s1 = s1 + a(i)
        s2 = s2 + a(i+1)
        s3 = s3 + a(i+2)
        s4 = s4 + a(i+3)
        s5 = s5 + a(i+4)
        s6 = s6 + a(i+5)
        s7 = s7 + a(i+6)
        s8 = s8 + a(i+7)
     end do
     i = i0 + 8*ng
     if (m >= 8*ng+1_8) s1 = s1 + a(i)
     if (m >= 8*ng+2_8) s2 = s2 + a(i+1)
     if (m >= 8*ng+3_8) s3 = s3 + a(i+2)
     if (m >= 8*ng+4_8) s4 = s4 + a(i+3)
     if (m >= 8*ng+5_8) s5 = s5 + a(i+4)
     if (m >= 8*ng+6_8) s6 = s6 + a(i+5)
     if (m >= 8*ng+7_8) s7 = s7 + a(i+6)
     cs(k) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
  end do
!$omp barrier
  if (tid == 0) then
     do k = 1, nc-1
        cs(k) = cs(k) + cs(k-1)
     end do
  end if
!$omp barrier
  do k = tid, nc-1, nt
     i0 = k*csize + 1
     m  = min(csize, n - k*csize + 1)
     ng = m / 8
     i  = i0
     s1 = cs(k-1)
     s2 = 0.0d0; s3 = 0.0d0; s4 = 0.0d0
     s5 = 0.0d0; s6 = 0.0d0; s7 = 0.0d0; s8 = 0.0d0
     do i = i0, i0 + 8*ng - 8, 8
        s1 = s1 + a(i)
        s2 = s2 + a(i+1)
        s3 = s3 + a(i+2)
        s4 = s4 + a(i+3)
        s5 = s5 + a(i+4)
        s6 = s6 + a(i+5)
        s7 = s7 + a(i+6)
        s8 = s8 + a(i+7)
        t = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
        v = t - a(i+1) - a(i+2) - a(i+3) - a(i+4) - a(i+5) - a(i+6) - a(i+7)
        b(i)   = v
        b(i+1) = v + a(i+1)
        b(i+2) = b(i+1) + a(i+2)
        b(i+3) = b(i+2) + a(i+3)
        b(i+4) = b(i+3) + a(i+4)
        b(i+5) = b(i+4) + a(i+5)
        b(i+6) = b(i+5) + a(i+6)
        b(i+7) = b(i+6) + a(i+7)
     end do
     t = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
     i = i0 + 8*ng
     do j = 1_8, m - 8*ng
        t = t + a(i)
        b(i) = t
        i = i + 1
     end do
  end do
!$omp end parallel
end subroutine tsvc_2_s3112_fp64
