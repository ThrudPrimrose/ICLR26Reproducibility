subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  !$omp parallel
  do j = 2, LEN_2D
    !$omp do simd schedule(static) simdlen(8) nowait
    do i = 1, LEN_2D
      aa(i, j) = aa(i, j - 1) + bb(i, j)
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s231_fp64
