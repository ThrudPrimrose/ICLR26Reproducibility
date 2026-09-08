! TSVC_2 quasi_affine_reduce_odd: out(1) = sum(a(i+1), i=1..LEN_1D-1, step 2)  [C 0-based]
! i.e. the even-position (1-based) elements: a(2), a(4), ..., a(2*n), n = LEN_1D/2.
! Eight independent accumulation chains: memory-bound, latency hiding via MLP.
subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(c, name="quasi_affine_reduce_odd_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_1d
  real(c_double) :: a(*)
  real(c_double), intent(out) :: out(1)
  integer(c_int64_t) :: n, n8, k
  real(c_double) :: s0, s1, s2, s3, s4, s5, s6, s7

  s0 = 0.0d0; s1 = 0.0d0; s2 = 0.0d0; s3 = 0.0d0
  s4 = 0.0d0; s5 = 0.0d0; s6 = 0.0d0; s7 = 0.0d0

  n = len_1d / 2
  n8 = n / 8
  do k = 1, n8
    s0 = s0 + a(16*k - 14)
    s1 = s1 + a(16*k - 12)
    s2 = s2 + a(16*k - 10)
    s3 = s3 + a(16*k - 8)
    s4 = s4 + a(16*k - 6)
    s5 = s5 + a(16*k - 4)
    s6 = s6 + a(16*k - 2)
    s7 = s7 + a(16*k)
  end do
  do k = 8*n8 + 1, n
    s0 = s0 + a(2*k)
  end do
  out(1) = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
end subroutine quasi_affine_reduce_odd_fp64
