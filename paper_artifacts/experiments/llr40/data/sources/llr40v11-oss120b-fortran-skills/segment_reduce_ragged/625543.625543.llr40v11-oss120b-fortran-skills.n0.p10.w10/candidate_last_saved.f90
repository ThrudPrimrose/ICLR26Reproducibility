subroutine segment_reduce_ragged_fp64(val, row_ptr, w, out, NSEG, workspace, workspace_size) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: NSEG
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: val(*)
  integer(c_int64_t), intent(in), target :: row_ptr(*)
  real(c_double), intent(in) :: w(*)
  real(c_double), intent(inout) :: out(*)
  integer(c_int64_t) :: s, e
  real(c_double) :: acc, acc2
  ! Debug pointers
  ! print *, 'DEBUG: C address of row_ptr', c_loc(row_ptr)
      print *, 'Debug: row_ptr(1)', row_ptr(1), 'row_ptr(2)', row_ptr(2)

  ! serial loop
  do s = 1, NSEG
    if (s == 1_c_int64_t) then
      print *, 'Debug: s=1 start', row_ptr(s) + 1_c_int64_t, 'end', row_ptr(s+1)
      print *, 'val(1)', val(1), 'w(1)', w(1)
    end if
    acc = 0.0_c_double
    do e = row_ptr(s) + 1_c_int64_t, row_ptr(s+1)
      acc = acc + val(e) * w(e)
        if (s == 1_c_int64_t .and. e <= 5_c_int64_t) then
          print *, 'e', e, 'val(e)', val(e), 'w(e)', w(e)
        end if
    end do
    out(s) = acc
        if (s == 1_c_int64_t) then
          acc2 = sum(val(row_ptr(s) + 1_c_int64_t:row_ptr(s+1)) * w(row_ptr(s) + 1_c_int64_t:row_ptr(s+1)))
          print *, 'Kernel acc', acc, 'acc2 sum', acc2
        end if
  end do
! end parallel
! End of loop (OpenMP parallel)
end subroutine segment_reduce_ragged_fp64
