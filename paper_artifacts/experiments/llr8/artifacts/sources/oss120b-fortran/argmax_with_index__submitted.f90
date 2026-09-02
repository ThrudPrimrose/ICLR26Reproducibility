! Argmax with index kernel optimized for benchmark using a single-pass loop
! C interoperable function name with fp64 suffix as required by the harness.

subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_bytes) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_1D
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t) :: i
  real(c_double) :: max_val
  integer(c_int64_t) :: max_idx

  ! Initialize with first element
  max_val = a(1)
  max_idx = 0

  do i = 2, LEN_1D
    if (a(i) > max_val) then
      max_val = a(i)
      max_idx = i - 1
    end if
  end do

  out_value = max_val
  out_index = max_idx
end subroutine argmax_with_index_fp64
