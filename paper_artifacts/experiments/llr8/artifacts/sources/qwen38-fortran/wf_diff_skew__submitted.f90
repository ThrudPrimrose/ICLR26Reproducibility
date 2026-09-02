subroutine wf_diff_skew_fp64(a, LEN_2D, workspace, workspace_size) bind(C, name="wf_diff_skew_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout), target :: a(*)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: n, i, j
  real(c_double), pointer, contiguous :: row(:), prev(:)
  n = LEN_2D
  if (n < 2) return
  !$omp parallel num_threads(4) default(none) shared(a,n,row,prev) private(i,j)
  do i = 1, n-1
     row  => a(i*n + 1 : i*n + n)
     prev => a((i-1)*n + 1 : (i-1)*n + n)
     !$omp do
     do j = 1, n-1
        row(j) = row(j) + prev(j) + prev(j+1)
     end do
     !$omp end do
  end do
  !$omp end parallel
end subroutine wf_diff_skew_fp64
