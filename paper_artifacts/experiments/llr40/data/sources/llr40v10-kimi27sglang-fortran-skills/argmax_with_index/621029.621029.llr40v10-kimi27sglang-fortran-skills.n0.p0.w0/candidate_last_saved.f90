subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: first, n

  n = LEN_1D

  if (n <= 0) then
    out_value(1) = 0.0d0
    out_index(1) = 0
    return
  end if

  first = int(maxloc(a, dim=1, kind=c_int64_t), c_int64_t)
  out_value(1) = a(first)
  out_index(1) = first

end subroutine argmax_with_index_fp64
