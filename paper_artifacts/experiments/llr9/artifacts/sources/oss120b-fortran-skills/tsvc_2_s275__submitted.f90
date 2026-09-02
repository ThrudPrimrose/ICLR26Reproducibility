subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  integer(c_int64_t), value, intent(in) :: workspace_size
  type(c_ptr), value, intent(in) :: workspace
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int64_t) :: i, j
  real(c_double), volatile :: tmp

  ! Serial loop (no OpenMP)
  do i = 1, len_2d
    if (aa(i, 1) > 0d0) then
      do j = 2, len_2d
        tmp = bb(i, j) * cc(i, j)
          aa(i, j) = aa(i, j-1) + tmp
      end do
    end if
  end do

end subroutine tsvc_2_s275_fp64
