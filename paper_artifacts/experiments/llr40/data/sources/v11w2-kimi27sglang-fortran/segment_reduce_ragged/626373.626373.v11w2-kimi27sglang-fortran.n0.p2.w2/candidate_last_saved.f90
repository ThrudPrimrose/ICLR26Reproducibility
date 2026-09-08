module segment_reduce_ragged_mod
  use iso_c_binding
  implicit none
contains
  subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, workspace, workspace_bytes) bind(C, name="segment_reduce_ragged_fp64")
    integer(c_int64_t), value, intent(in) :: NSEG
    real(c_double), intent(out) :: out(0:NSEG - 1)
    integer(c_int64_t), intent(in) :: row_ptr(0:NSEG)
    real(c_double), intent(in) :: val(0:NSEG * 24 - 1)
    real(c_double), intent(in) :: w(0:NSEG * 24 - 1)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_bytes
    integer(c_int64_t) :: s, e
    real(c_double) :: acc
    !$omp parallel do schedule(guided, 16) private(s, e, acc)
    do s = 0_c_int64_t, NSEG - 1_c_int64_t
      acc = 0.0_c_double
      !$omp simd reduction(+:acc)
      do e = row_ptr(s), row_ptr(s + 1_c_int64_t) - 1_c_int64_t
        acc = acc + val(e) * w(e)
      end do
      out(s) = acc
    end do
  end subroutine segment_reduce_ragged_fp64
end module segment_reduce_ragged_mod
