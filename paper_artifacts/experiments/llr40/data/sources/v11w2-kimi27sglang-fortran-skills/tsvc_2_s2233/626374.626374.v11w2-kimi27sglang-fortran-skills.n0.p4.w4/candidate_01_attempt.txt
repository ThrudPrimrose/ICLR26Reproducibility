subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int64_t) :: i, j

  !$omp parallel do schedule(static)
  do i = 9, len_2d
    do j = 9, len_2d
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
  end do
  !$omp end parallel do

  !$omp parallel do schedule(static)
  do j = 9, len_2d
    do i = 9, len_2d
      bb(j, i) = bb(j, i - 1) + cc(j, i)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s2233_fp64
