subroutine s3111(LEN_1D, a, b) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(2)
  integer :: i
  real(c_double) :: sum_val
  sum_val = 0.0d0
  !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(+:sum_val)
  do i = 1, LEN_1D
    if (a(i) > 0.0d0) then
      sum_val = sum_val + a(i)
    end if
  end do
  !$omp end parallel do
  b(1) = sum_val
  b(2) = 0.0d0
end subroutine s3111

subroutine tsvc_2_s3111_fp64(a, b, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  ! Scalars first as per ABI requirements
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  ! Arrays
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(2)
  integer(c_int64_t) :: i
  real(c_double) :: sum_val
  sum_val = 0.0d0
  !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(+:sum_val)
  do i = 1, LEN_1D
    if (a(i) > 0.0d0) then
      sum_val = sum_val + a(i)
    end if
  end do
  !$omp end parallel do
  b(1) = sum_val
end subroutine tsvc_2_s3111_fp64
