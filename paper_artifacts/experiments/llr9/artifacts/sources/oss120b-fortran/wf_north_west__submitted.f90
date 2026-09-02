subroutine wf_north_west_fp64(a, len_2d, workspace, workspace_bytes) bind(C, name="wf_north_west_fp64")
  use iso_c_binding
  type(c_ptr), value :: a
  integer(c_int64_t), value :: len_2d
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes

  real(c_double), pointer :: arr(:)
  real(c_double), pointer :: row(:), prev_row(:)
  integer(c_int64_t) :: n
  integer :: n_int
  integer :: i_int, j_int, offset_int, prev_offset_int, idx_int, prev_idx_int
  type(c_ptr) :: rp, pp
  !

  n = len_2d
  n_int = int(len_2d)

  call c_f_pointer(a, arr, [len_2d*len_2d])

  do i_int = 1, n_int - 1
    offset_int = i_int * n_int
    prev_offset_int = offset_int - n_int
    !
    !
    rp = c_loc(arr(offset_int + 1))
    pp = c_loc(arr(prev_offset_int + 1))
    call c_f_pointer(rp, row, [n_int])
    call c_f_pointer(pp, prev_row, [n_int])
    do j_int = 2, n_int
      row(j_int) = row(j_int) + prev_row(j_int) + row(j_int - 1)
    end do
  end do

end subroutine wf_north_west_fp64
