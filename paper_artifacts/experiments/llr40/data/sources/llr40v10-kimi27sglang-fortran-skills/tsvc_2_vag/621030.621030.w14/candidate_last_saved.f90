subroutine tsvc_2_vag_fp64(a, b, ip, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  real(c_double), intent(out) :: a(n)
  real(c_double), intent(in) :: b(n)
  integer(c_int32_t), intent(in) :: ip(n)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i

  !$omp single
  write(*,'(A,I0)') 'n=', n
  !$omp end single

  !$omp parallel do simd schedule(static)
  do i = 1, n
    a(i) = b(ip(i))
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vag_fp64
