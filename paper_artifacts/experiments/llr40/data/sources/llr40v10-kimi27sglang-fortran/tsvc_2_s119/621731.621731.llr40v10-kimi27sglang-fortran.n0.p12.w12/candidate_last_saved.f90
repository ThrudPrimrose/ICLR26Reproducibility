subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(c, name="tsvc_2_s119_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(in) :: bb(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: i, j

  do i = 1, LEN_2D-1
    !$omp parallel do simd private(j)
    do j = 1, LEN_2D-1
      aa(j,i) = aa(j-1,i-1) + bb(j,i)
    end do
    !$omp end parallel do simd
  end do
end subroutine tsvc_2_s119_fp64
