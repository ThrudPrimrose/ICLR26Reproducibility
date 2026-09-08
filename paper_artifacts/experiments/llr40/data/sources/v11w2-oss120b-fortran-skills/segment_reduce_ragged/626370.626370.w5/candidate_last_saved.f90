module segment_reduce_ragged_mod
  use iso_c_binding, only: c_int64_t, c_double, c_ptr
  implicit none
contains
  subroutine segment_reduce_ragged_fp64(row_ptr, val, w, out, NSEG, workspace, workspace_size) bind(C)
    ! Arguments from C
    integer(c_int64_t), intent(in) :: row_ptr(*)
    real(c_double), intent(in) :: val(*)
    real(c_double), intent(in) :: w(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: NSEG
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size

    interface
      subroutine segment_reduce_ragged_fp64_impl(row_ptr, val, w, out, NSEG, workspace, workspace_size) bind(C, name="segment_reduce_ragged_fp64_impl")
        use iso_c_binding
        integer(c_int64_t), intent(in) :: row_ptr(*)
        real(c_double), intent(in) :: val(*)
        real(c_double), intent(in) :: w(*)
        real(c_double), intent(out) :: out(*)
        integer(c_int64_t), value, intent(in) :: NSEG
        type(c_ptr), value, intent(in) :: workspace
        integer(c_int64_t), value, intent(in) :: workspace_size
      end subroutine segment_reduce_ragged_fp64_impl
    end interface

    call segment_reduce_ragged_fp64_impl(row_ptr, val, w, out, NSEG, workspace, workspace_size)
  end subroutine segment_reduce_ragged_fp64
end module segment_reduce_ragged_mod
