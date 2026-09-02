subroutine tsvc_2_s231_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j, tid, nt, istart, iend

  if (len_2d < 256_c_int64_t) then
    do j = 2, len_2d
      !$omp simd
      do i = 1, len_2d
        aa(i, j) = aa(i, j - 1) + bb(i, j)
      end do
      !$omp end simd
    end do
  else
    nt = omp_get_max_threads()
    !$omp parallel private(i, j, tid, istart, iend)
    tid = omp_get_thread_num()
    istart = (len_2d * tid) / nt + 1
    iend = (len_2d * (tid + 1)) / nt
    do j = 2, len_2d
      !$omp simd
      do i = istart, iend
        aa(i, j) = aa(i, j - 1) + bb(i, j)
      end do
      !$omp end simd
    end do
    !$omp end parallel
  end if
end subroutine tsvc_2_s231_fp64
