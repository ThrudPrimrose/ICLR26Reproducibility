subroutine segment_reduce_ragged(row_ptr, val, w, out, NSEG) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: NSEG
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t) :: s, e
  real(c_double) :: acc

  !$omp parallel do schedule(dynamic) default(none) &
  !$omp& shared(row_ptr, val, w, out, NSEG) private(e, acc)
  do s = 1, NSEG
    acc = 0.0d0
    !$omp simd
    do e = row_ptr(s) + 1, row_ptr(s+1)
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
  !$omp end parallel do
end subroutine segment_reduce_ragged
