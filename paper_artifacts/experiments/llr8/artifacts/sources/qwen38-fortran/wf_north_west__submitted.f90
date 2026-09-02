subroutine wf_north_west_fp64(a, LEN_2D) bind(C, name="wf_north_west_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D*LEN_2D)
  integer(c_int64_t) :: i, j, q
  if (LEN_2D < 2) return
  do i = 1, LEN_2D-1
    do j = 1, LEN_2D-1
      q = i*LEN_2D + j
      a(q+1) = (a(q+1) + a(q+1-LEN_2D)) + a(q)
    end do
  end do
end subroutine
