subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, workspace, workspace_size) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: NSEG
  real(c_double), intent(out) :: out(*)
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: s
  integer(c_int64_t) :: start, finish, e
  real(c_double) :: acc

  ! Serial loop
  do s = 1, NSEG
    acc = 0.0_c_double
    start = row_ptr(s) + 1
    finish = row_ptr(s+1)
    do e = start, finish
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
  ! End of serial loop
end subroutine segment_reduce_ragged_fp64
