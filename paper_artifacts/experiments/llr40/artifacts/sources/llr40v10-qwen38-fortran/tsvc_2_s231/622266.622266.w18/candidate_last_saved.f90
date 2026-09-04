subroutine tsvc_2_s231_fp64(aa, bb, len2d) bind(C, name='tsvc_2_s231_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*)
  integer(c_int64_t), intent(in) :: len2d
  integer(c_int64_t) :: i, j
  !$omp parallel do schedule(static) private(i,j)
  do i = 0, len2d-1
    do j = 1, len2d-1
      aa(j*len2d + i + 1) = aa((j-1)*len2d + i + 1) + bb(j*len2d + i + 1)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s231_fp64
