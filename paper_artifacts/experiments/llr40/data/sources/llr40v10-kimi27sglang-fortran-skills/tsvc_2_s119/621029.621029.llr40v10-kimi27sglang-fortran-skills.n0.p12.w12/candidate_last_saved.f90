subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s119_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D * LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D * LEN_2D)

  integer(c_int64_t) :: i, j, n

  if (LEN_2D < 2) return
  n = LEN_2D

  !$omp parallel private(i, j)
  do i = 2, n
    !$omp do simd schedule(simd:static)
    do j = 2, n
      aa((i - 1) * n + j) = aa((i - 2) * n + (j - 1)) + bb((i - 1) * n + j)
    end do
    !$omp end do simd
  end do
  !$omp end parallel
end subroutine tsvc_2_s119_fp64
