subroutine segment_reduce_ragged_fp64(out, row_ptr, a, b, nseg) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: nseg
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(in) :: b(*)

  integer(c_int64_t) :: s, e
  real(c_double) :: acc

  do s = 1, nseg
    acc = 0.0d0
    do e = row_ptr(s) + 1, row_ptr(s+1)
      acc = acc + a(e) * b(e)
    end do
    out(s) = acc
  end do
end subroutine segment_reduce_ragged_fp64
