subroutine wf_triangular_fp64(a, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n, n)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i, j, t, i_start, i_end
  ! Wavefront triangular update for j >= i using wavefront parallelism
  !$omp parallel private(i, j, t, i_start, i_end)
  do t = 4, 2*n
    i_start = t - n
    if (i_start < 2_c_int64_t) i_start = 2_c_int64_t
    i_end = t / 2
    if (i_end > n) i_end = n
    !$omp do schedule(static)
    do i = i_start, i_end
      j = t - i
      a(j, i) = a(j, i) + a(j, i-1) + a(j-1, i)
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wf_triangular_fp64
