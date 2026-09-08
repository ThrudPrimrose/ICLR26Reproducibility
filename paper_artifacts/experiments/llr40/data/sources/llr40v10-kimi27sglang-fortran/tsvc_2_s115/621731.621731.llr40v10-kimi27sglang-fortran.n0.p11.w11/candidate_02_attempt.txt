subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(0:LEN_2D-1)
  real(c_double), intent(in) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: i, j, jm1
  real(c_double) :: aj0, aj1

  if (LEN_2D .le. 0) return

  aj0 = a(0)
  do i = 1, LEN_2D-1
    a(i) = a(i) - aa(i, 0) * aj0
  end do

  jm1 = LEN_2D - 1
  do j = 1, jm1 - 1, 2
    aj0 = a(j)
    aj1 = a(j+1) - aa(j+1, j) * aj0
    a(j+1) = aj1
    do i = j+2, LEN_2D-1
      a(i) = a(i) - aa(i, j) * aj0 - aa(i, j+1) * aj1
    end do
  end do

  if (mod(jm1, 2_c_int64_t) .ne. 0) then
    j = jm1 - 1
    aj0 = a(j)
    do i = j+1, LEN_2D-1
      a(i) = a(i) - aa(i, j) * aj0
    end do
  end if
end subroutine tsvc_2_s115_fp64
