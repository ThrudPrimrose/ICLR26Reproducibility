module tsvc_2_s1244_m
  use, intrinsic :: iso_c_binding
  implicit none
  integer, parameter :: BLK = 2048
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(c)
    type(c_ptr), value :: a, b, c, d
    integer(c_int64_t), value :: LEN_1D
    real(c_double), pointer, contiguous :: ap(:), bp(:), cp(:), dp(:)
    real(c_double) :: tmp(BLK)
    integer(c_int64_t) :: bs, be, m, k, n
    n = LEN_1D - 1
    if (n < 1) return
    call c_f_pointer(a, ap, [n+1])
    call c_f_pointer(b, bp, [n+1])
    call c_f_pointer(c, cp, [n+1])
    call c_f_pointer(d, dp, [n+1])
    do bs = 1, n, BLK
       be = min(n, bs + BLK - 1)
       m = be - bs + 1
       !$omp simd
       do k = 1, m
          tmp(k) = ap(bs + k)
       end do
       !$omp simd aligned(ap,bp,cp,dp:64)
       do k = 1, m
          ap(bs + k - 1) = bp(bs + k - 1) + cp(bs + k - 1)*cp(bs + k - 1) &
                         + bp(bs + k - 1)*bp(bs + k - 1) + cp(bs + k - 1)
          dp(bs + k - 1) = ap(bs + k - 1) + tmp(k)
       end do
    end do
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_m
