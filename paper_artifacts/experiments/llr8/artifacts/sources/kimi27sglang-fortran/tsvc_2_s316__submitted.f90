subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C, name="tsvc_2_s316_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value   :: LEN_1D
  real(c_double), intent(in)  :: a(LEN_1D)
  real(c_double), intent(out) :: result(1)
  integer(c_int64_t) :: i
  real(c_double) :: x

  if (LEN_1D <= 0) return
  x = a(1)
  if (LEN_1D > 1) then
     !$omp parallel do reduction(min:x)
     do i = 2, LEN_1D
        x = min(x, a(i))
     end do
     !$omp end parallel do
  end if
  result(1) = x
end subroutine tsvc_2_s316_fp64
