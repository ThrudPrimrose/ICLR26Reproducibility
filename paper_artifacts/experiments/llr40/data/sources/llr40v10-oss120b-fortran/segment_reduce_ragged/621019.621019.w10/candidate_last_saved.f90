!$GCC$ push_options
!$GCC$ optimize ("O0")
subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, NSEG) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  integer(C_INT64_T), intent(in) :: row_ptr(*)
  real(C_DOUBLE), intent(in) :: val(*)
  real(C_DOUBLE), intent(in) :: w(*)
  real(C_DOUBLE), intent(out) :: out(*)
  integer(C_INT64_T), value :: NSEG
  integer(C_INT64_T) :: s, e, start_idx, end_idx
  real(C_DOUBLE) :: acc
  do s = 1, NSEG
    start_idx = row_ptr(s) + 1
    end_idx = row_ptr(s+1)
    acc = 0.0d0
        do e = start_idx, end_idx
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
end subroutine segment_reduce_ragged_fp64
!$GCC$ pop_options
