module segment_reduce_ragged_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, NSEG) bind(C, name="segment_reduce_ragged_fp64")
    integer(c_int64_t), intent(in) :: row_ptr(0:*)
    real(c_double), intent(in) :: val(0:*), w(0:*)
    real(c_double), intent(out) :: out(0:*)
    integer(c_int64_t), value :: NSEG
    integer(c_int64_t) :: s, e
    real(c_double) :: acc
    do s = 0, NSEG-1
        acc = 0.0_c_double
        do e = row_ptr(s), row_ptr(s+1)-1
            acc = acc + val(e) * w(e)
        end do
        out(s) = acc
    end do
  end subroutine segment_reduce_ragged_fp64
end module segment_reduce_ragged_mod
