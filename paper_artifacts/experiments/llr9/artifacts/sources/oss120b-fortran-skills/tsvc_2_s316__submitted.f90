subroutine tsvc_2_s316_fp64(a, result, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: result(1)
  integer(c_int64_t) :: i
  real(c_double) :: min_val

  ! Initialise the reduction variable with the first element.
  if (LEN_1D > 0) then
    min_val = a(1)
  else
    min_val = 0.0d0
  end if

  !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(min:min_val)
  do i = 2, LEN_1D
    if (a(i) < min_val) min_val = a(i)
  end do
  !$omp end parallel do

  result(1) = min_val
end subroutine tsvc_2_s316_fp64
