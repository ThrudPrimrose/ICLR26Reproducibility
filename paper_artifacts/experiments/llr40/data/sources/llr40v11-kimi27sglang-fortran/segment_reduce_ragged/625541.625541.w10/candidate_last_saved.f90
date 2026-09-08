subroutine segment_reduce_ragged_fp64(val, row_ptr, w, out, nseg, workspace, workspace_bytes) bind(c, name='segment_reduce_ragged_fp64')
  use iso_c_binding, only: c_int64_t, c_double, c_int8_t
  use iso_fortran_env, only: output_unit
  implicit none
  integer(c_int64_t), value, intent(in) :: nseg
  integer(c_int64_t), intent(in) :: row_ptr(*)
  real(c_double), intent(in) :: val(*)
  real(c_double), intent(in) :: w(*)
  real(c_double), intent(out) :: out(*)
  integer(c_int8_t), intent(in) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_bytes
  integer(c_int64_t) :: s, e
  real(c_double) :: acc
  write(output_unit, *) 'enter nseg=', nseg, 'total=', row_ptr(nseg+1)
  flush(output_unit)
  !$omp parallel do schedule(dynamic, 1024) private(s, e, acc)
  do s = 1_c_int64_t, nseg
    acc = 0.0_c_double
    do e = row_ptr(s) + 1_c_int64_t, row_ptr(s + 1)
      acc = acc + val(e) * w(e)
    end do
    out(s) = acc
  end do
  !$omp end parallel do
  write(output_unit, *) 'exit'
  flush(output_unit)
end subroutine segment_reduce_ragged_fp64
