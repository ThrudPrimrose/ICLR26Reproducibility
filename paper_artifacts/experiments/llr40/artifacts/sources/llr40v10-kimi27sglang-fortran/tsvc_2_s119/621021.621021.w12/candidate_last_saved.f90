subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  do i = 2, LEN_2D
    do j = 2, LEN_2D
      aa(j, i) = aa(j - 1, i - 1) + bb(j, i)
    end do
  end do
end subroutine tsvc_2_s119_fp64
