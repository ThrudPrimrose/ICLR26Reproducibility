subroutine scatter_accum_dup_fp64(bins, ip, src, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  real(c_double), intent(inout) :: bins(n)
  integer(c_int32_t), intent(in) :: ip(n)
  real(c_double), intent(in) :: src(n)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: i
  print *, 'nt=', omp_get_max_threads(), 'n=', n, 'ws=', workspace_size
  do i = 1, n
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
end subroutine scatter_accum_dup_fp64
