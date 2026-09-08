subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding, only: c_double, c_int64_t, c_int32_t, c_ptr
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do simd default(none) schedule(static) shared(a,b,ip,LEN_1D) private(i)
    do i = 1, LEN_1D
    a(i) = a(i) + 2.0d0 * b(ip(i))
  end do
  
end subroutine tsvc_2_s4112_fp64
