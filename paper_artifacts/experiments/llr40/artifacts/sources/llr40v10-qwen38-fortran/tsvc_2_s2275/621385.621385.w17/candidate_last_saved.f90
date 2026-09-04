! TSVC tsvc_2 kernel s2275 (single-invocation, fp64) -- optimized Fortran.
!
! Reference semantics (element-independent, so order does not matter):
!   aa[j*N+i] = aa[j*N+i] + bb[j*N+i]*cc[j*N+i]   for all i,j   (all N*N elements)
!   a[i]      = b[i] + c[i]*d[i]                   for all i
! The reference walks the 2D panel column by column (stride N); here the same
! per-element update is issued as one flat contiguous pass, which vectorizes
! at full width and parallelizes across cores.

subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C, name='tsvc_2_s2275_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value          :: a, aa, b, bb, c, cc, d
  integer(c_int64_t), value   :: len_2d

  real(c_double), dimension(:), pointer :: fa, faa, fb, fbb, fc, fcc, fd
  integer(c_int64_t) :: n, nn, k

  n  = len_2d
  nn = n * n

  call c_f_pointer(a,  fa,  [n])
  call c_f_pointer(aa, faa, [nn])
  call c_f_pointer(b,  fb,  [n])
  call c_f_pointer(bb, fbb, [nn])
  call c_f_pointer(c,  fc,  [n])
  call c_f_pointer(cc, fcc, [nn])
  call c_f_pointer(d,  fd,  [n])

  ! Main body: single contiguous elementwise FMA over all N*N elements.
  !$omp parallel do schedule(static)
  do k = 1, nn
    faa(k) = faa(k) + fbb(k) * fcc(k)
  end do
  !$omp end parallel do

  ! One-panel FMA, independent elementset; separate small parallel loop.
  !$omp parallel do schedule(static)
  do k = 1, n
    fa(k) = fb(k) + fc(k) * fd(k)
  end do
  !$omp end parallel do
end subroutine tsvc_2_s2275_fp64
