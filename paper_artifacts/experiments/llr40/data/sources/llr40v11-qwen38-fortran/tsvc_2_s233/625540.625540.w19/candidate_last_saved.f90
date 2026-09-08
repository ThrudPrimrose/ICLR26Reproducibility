subroutine tsvc_2_s233_fp64(aa, bb, cc, n) bind(C, name="tsvc_2_s233_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t) :: i, j
  do i = 8, n-1
    do j = 8, n-1
      aa(j*n + i + 1) = aa((j-1)*n + i + 1) + cc(j*n + i + 1)
    end do
    do j = 8, n-1
      bb(j*n + i + 1) = bb(j*n + i - 1 + 1) + cc(j*n + i + 1)
    end do
  end do
end subroutine tsvc_2_s233_fp64
