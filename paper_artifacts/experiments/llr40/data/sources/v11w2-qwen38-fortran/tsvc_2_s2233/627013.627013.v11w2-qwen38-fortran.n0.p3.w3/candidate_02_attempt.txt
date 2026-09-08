subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C, name='tsvc_2_s2233_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), value :: len_2d
  integer(c_int64_t) :: i, j, n
  n = len_2d
  do i = 8, n - 1
     do j = 8, n - 1
        aa(j*n + i + 1) = aa((j-1)*n + i + 1) + cc(j*n + i + 1)
     end do
     do j = 8, n - 1
        bb(i*n + j + 1) = bb((i-1)*n + j + 1) + cc(i*n + j + 1)
     end do
  end do
end subroutine
