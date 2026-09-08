subroutine versioned_distance_update_fp64(a, b, c, LEN_1D, K, workspace, workspace_bytes) bind(c, name='versioned_distance_update_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*), c(*)
  integer(c_int64_t), value :: LEN_1D, K, workspace_bytes
  type(c_ptr), value :: workspace
  integer(c_int64_t) :: i
  write(*,'(A,2I12)') 'DEBUG LEN K ', LEN_1D, K
  do i = K+1, LEN_1D
    a(i) = 0.75_c_double * a(i-K) + b(i) * c(i)
  end do
  write(*,'(A,3ES24.15)') 'DEBUG a0 aK aN ', a(1), a(K+1), a(LEN_1D)
end subroutine versioned_distance_update_fp64
