subroutine segment_reduce_ragged_fp64(val, row_ptr, w, out, nseg, workspace, workspace_size) bind(C, name='segment_reduce_ragged_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: nseg, workspace_size
  integer(c_int64_t), intent(in) :: row_ptr(nseg + 1)
  real(c_double), intent(in) :: val(nseg * 24), w(nseg * 24)
  real(c_double), intent(out) :: out(nseg)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: s, e
  real(c_double) :: acc

  !$omp parallel do schedule(dynamic, 1) private(acc, e)
  do s = 1, nseg
    acc = 0.0d0
    do e = row_ptr(s), row_ptr(s + 1) - 1
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
  !$omp end parallel do
end subroutine segment_reduce_ragged_fp64
