subroutine s2710(a, b, c, d, e, x, len_1d)
  use iso_c_binding
  implicit none
  integer(C_INT64_T), intent(in) :: len_1d
  real(8), intent(inout) :: a(len_1d), b(len_1d), c(len_1d)
  real(8), intent(in) :: d(len_1d), e(len_1d)
  real(8), intent(in) :: x(len_1d)
  integer :: i
  logical :: x_gt0
  logical :: len_gt10

  x_gt0 = (x(1) > 0.0d0)
  len_gt10 = (len_1d > 10)

!$omp parallel private(i)
!$omp do
  do i = 1, len_1d
    if (a(i) > b(i)) then
      a(i) = a(i) + b(i) * d(i)
      if (len_gt10) then
        c(i) = c(i) + d(i) * d(i)
      else
        c(i) = d(i) * e(i) + 1.0d0
      end if
    else
      b(i) = a(i) + e(i) * e(i)
      if (x_gt0) then
        c(i) = a(i) + d(i) * d(i)
      else
        c(i) = c(i) + e(i) * e(i)
      end if
    end if
  end do
!$omp end do
!$omp end parallel
end subroutine s2710

subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d, workspace, workspace_bytes) bind(C, name="tsvc_2_s2710_fp64")
  use iso_c_binding
  implicit none
  real(C_DOUBLE), dimension(*), intent(inout) :: a
  real(C_DOUBLE), dimension(*), intent(inout) :: b
  real(C_DOUBLE), dimension(*), intent(inout) :: c
  real(C_DOUBLE), dimension(*), intent(in) :: d
  real(C_DOUBLE), dimension(*), intent(in) :: e
  real(C_DOUBLE), dimension(*), intent(in) :: x
  integer(C_INT64_T), value :: len_1d
  type(C_PTR), value :: workspace
  integer(C_INT64_T), value :: workspace_bytes

  call s2710(a, b, c, d, e, x, len_1d)
end subroutine tsvc_2_s2710_fp64
