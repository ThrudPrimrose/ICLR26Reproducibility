subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D, workspace, workspace_bytes) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(out) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int), intent(in) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i

  ! Vectorized gather using ip (0-based) to index b (1-based Fortran indexing)
  !$omp simd
  do i = 1, LEN_1D
    a(i) = b(ip(i))
  end do
end subroutine tsvc_2_vag_fp64
