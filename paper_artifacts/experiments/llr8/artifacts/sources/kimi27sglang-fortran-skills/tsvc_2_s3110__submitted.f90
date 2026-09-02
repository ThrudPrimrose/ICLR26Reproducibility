subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name='tsvc_2_s3110_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), target, intent(in) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(2, 2)

  integer :: n, k
  integer(c_int64_t) :: idx, huge64
  real(c_double) :: maxv
  real(c_double), pointer, contiguous :: a1(:)

  n = int(LEN_2D)
  call c_f_pointer(c_loc(aa), a1, [n * n])

  maxv = a1(1)
  !$omp simd reduction(max:maxv)
  do k = 2, n * n
    maxv = max(maxv, a1(k))
  end do
  !$omp end simd

  huge64 = huge(1_c_int64_t)
  idx = huge64
  !$omp simd reduction(min:idx)
  do k = 1, n * n
    idx = min(idx, merge(int(k - 1, c_int64_t), huge64, a1(k) >= maxv))
  end do
  !$omp end simd

  bb(1, 1) = maxv + dble(idx / int(n, c_int64_t) + mod(idx, int(n, c_int64_t)))
end subroutine tsvc_2_s3110_fp64
