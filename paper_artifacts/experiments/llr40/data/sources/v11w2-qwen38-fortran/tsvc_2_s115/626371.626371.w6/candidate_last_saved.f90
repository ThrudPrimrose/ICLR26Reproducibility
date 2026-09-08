subroutine tsvc_2_s115_fp64(a, aa, len2d) bind(c, name="tsvc_2_s115_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: len2d
  real(c_double), intent(inout) :: a(len2d)
  real(c_double), intent(in) :: aa(len2d*len2d)
  integer(c_int64_t) :: j, i
  real(c_double) :: t
  do j = 0, len2d-1
    do i = j+1, len2d-1
      t = aa(j*len2d + i + 1) * a(j+1)
      a(i+1) = a(i+1) - t
    end do
  end do
end subroutine
