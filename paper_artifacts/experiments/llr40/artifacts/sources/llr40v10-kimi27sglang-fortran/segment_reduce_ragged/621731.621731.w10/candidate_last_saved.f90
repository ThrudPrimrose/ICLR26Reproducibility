subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, NSEG) bind(c, name="segment_reduce_ragged_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: NSEG
  integer(c_int64_t), intent(in) :: row_ptr(NSEG + 1)
  real(c_double), intent(in) :: val(NSEG * 24)
  real(c_double), intent(in) :: w(NSEG * 24)
  real(c_double), intent(out) :: out(NSEG)
  integer(c_int64_t) :: s, e, start, finish
  real(c_double) :: acc

  !$omp parallel do schedule(dynamic, 64) private(s, e, start, finish, acc)
  do s = 1, NSEG
    start = row_ptr(s) + 1
    finish = row_ptr(s + 1)
    acc = 0.0_c_double
    do e = start, finish
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
  !$omp end parallel do
end subroutine segment_reduce_ragged_fp64
