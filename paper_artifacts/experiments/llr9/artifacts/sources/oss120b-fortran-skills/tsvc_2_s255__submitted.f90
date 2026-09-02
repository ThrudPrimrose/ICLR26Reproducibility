subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double), parameter :: factor = 0.333d0
  ! Handle first element (wrap-around from end)
  if (LEN_1D >= 1) then
    a(1) = (b(1) + b(LEN_1D) + b(LEN_1D-1)) * factor
  end if
  ! Handle second element
  if (LEN_1D >= 2) then
    a(2) = (b(2) + b(1) + b(LEN_1D)) * factor
  end if
  ! Main loop for remaining elements
  if (LEN_1D > 2) then
    !$omp parallel do simd schedule(static) default(none) shared(a,b,LEN_1D) private(i)
    do i = 3, LEN_1D
      a(i) = (b(i) + b(i-1) + b(i-2)) * factor
    end do
  end if
end subroutine tsvc_2_s255_fp64
