subroutine tsvc_2_s1244_fp64(a, b, c, d, len_1d, workspace, workspace_bytes) bind(C, name="tsvc_2_s1244_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_1d
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*), c(*)
  real(c_double), intent(out) :: d(*)
  integer(c_int64_t) :: i
  real(c_double), allocatable :: a_old(:)

  if (len_1d <= 1) then
    return
  end if

  allocate(a_old(len_1d))
  !$omp parallel do private(i) schedule(static)
  do i = 1, len_1d
    a_old(i) = a(i)
  end do
  !$omp end parallel do

  !$omp parallel do private(i) schedule(static)
  do i = 1, len_1d-1
    a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
  end do
  !$omp end parallel do

  !$omp parallel do private(i) schedule(static)
  do i = 1, len_1d-1
    d(i) = a(i) + a_old(i+1)
  end do
  !$omp end parallel do

  deallocate(a_old)
end subroutine tsvc_2_s1244_fp64
