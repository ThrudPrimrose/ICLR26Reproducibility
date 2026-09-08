subroutine wf_triangular_fp64(a, LEN_2D, workspace, workspace_size) bind(C, name="wf_triangular_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), intent(inout) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, k
  integer(c_int64_t) :: i_low, i_high

  !$omp parallel default(none) shared(a, LEN_2D) private(i, j, k, i_low, i_high)
  do k = 4_c_int64_t, 2_c_int64_t * LEN_2D
    i_low = max(2_c_int64_t, k - LEN_2D)
    i_high = min(LEN_2D, k / 2_c_int64_t)
    !$omp do schedule(static)
    do i = i_low, i_high
      j = k - i
      a(j,i) = a(j,i) + a(j,i-1) + a(j-1,i)
    end do
    !$omp end do
  end do
  !$omp end parallel

end subroutine wf_triangular_fp64
