subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, nseg, workspace, workspace_size) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: out(*)
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  integer(c_int64_t), value, intent(in) :: nseg
  real(c_double), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: s, e, lo, hi
  real(c_double) :: acc

  !$omp parallel do schedule(static)
  do s = 1, nseg
    lo = row_ptr(s) + 1
    hi = row_ptr(s + 1)
    acc = 0.0d0
    do e = lo, hi
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
end subroutine segment_reduce_ragged_fp64
