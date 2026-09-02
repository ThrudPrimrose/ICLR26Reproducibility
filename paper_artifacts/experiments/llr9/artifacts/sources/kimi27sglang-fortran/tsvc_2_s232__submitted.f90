subroutine tsvc_2_s232_fp64(aa, bb, LEN_2D) bind(c, name='tsvc_2_s232_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(in) :: bb(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: j, i

  !$omp parallel do schedule(runtime) private(i)
  do j = 1, LEN_2D - 1
    do i = 1, j
      aa(i, j) = aa(i-1, j) * aa(i-1, j) + bb(i, j)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s232_fp64
