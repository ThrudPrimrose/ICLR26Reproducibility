subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(0:LEN_2D-1)
  real(c_double), intent(in) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: j
  integer :: n, i
  real(c_double) :: aj

  n = int(LEN_2D)
  do j = 0, n-1
    aj = a(j)
    do i = j+1, n-1
      a(i) = a(i) - aa(i, j) * aj
    end do
  end do
end subroutine tsvc_2_s115_fp64
