subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, ws, ws_len) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  integer(c_int64_t), value :: NSEG
  type(c_ptr), value :: ws
  integer(c_int64_t), value :: ws_len
  integer(c_int64_t) :: s
  integer(c_int64_t) :: start_idx, end_idx, e
  real(c_double) :: acc
    do s = 0, NSEG - 1
    start_idx = row_ptr(s+1)
    end_idx = row_ptr(s+2) - 1
    acc = 0.0_c_double
    !$omp simd reduction(+:acc)
    do e = start_idx, end_idx
      acc = acc + val(e+1) * w(e+1)
    end do
    out(s+1) = acc
  end do
  end subroutine segment_reduce_ragged_fp64
