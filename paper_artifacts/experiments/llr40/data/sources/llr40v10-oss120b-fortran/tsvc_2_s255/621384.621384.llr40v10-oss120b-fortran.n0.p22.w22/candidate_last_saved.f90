subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(C, name="tsvc_2_s255_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(out) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value :: len_1d
  real(c_double), parameter :: factor = 0.333_c_double
  integer(c_int64_t) :: i

  if (len_1d >= 1) then
    a(1) = (b(1) + b(len_1d) + b(len_1d-1)) * factor
  end if

  if (len_1d >= 2) then
    a(2) = (b(2) + b(1) + b(len_1d)) * factor
  end if

  if (len_1d >= 3) then
    !$omp parallel do
    do i = 3, len_1d
      a(i) = (b(i) + b(i-1) + b(i-2)) * factor
    end do
    !$omp end parallel do
  end if

end subroutine tsvc_2_s255_fp64
