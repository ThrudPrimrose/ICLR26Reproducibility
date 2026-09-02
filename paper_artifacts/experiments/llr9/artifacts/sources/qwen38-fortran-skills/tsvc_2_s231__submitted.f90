subroutine tsvc_2_s231_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C, name='tsvc_2_s231_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j
  integer :: nt, tid
  integer(c_int64_t) :: i_lo, i_hi
  nt = omp_get_max_threads()
  !$omp parallel private(tid, i_lo, i_hi, i, j)
  tid = omp_get_thread_num()
  i_lo = (len_2d * tid) / nt + 1
  i_hi = (len_2d * (tid + 1)) / nt
  do j = 2, len_2d
    do i = i_lo, i_hi
      aa(i, j) = aa(i, j-1) + bb(i, j)
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s231_fp64
