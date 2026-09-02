subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D, workspace, workspace_bytes) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  real(C_DOUBLE), dimension(*), intent(inout) :: aa
  real(C_DOUBLE), dimension(*), intent(in) :: bb, cc
  integer(C_INT64_T), value :: LEN_2D
  type(C_PTR), value :: workspace
  integer(C_INT64_T), value :: workspace_bytes

  integer :: i, j
  integer(C_INT64_T) :: idx, prev_idx
  real(C_DOUBLE), volatile :: prod

  !$omp parallel do schedule(static) default(none) private(i, j, idx, prev_idx, prod) shared(aa, bb, cc, LEN_2D)
  do i = 1, LEN_2D
    if (aa(i) > 0.0d0) then
      do j = 2, LEN_2D
        idx = (j - 1) * LEN_2D + i
        prev_idx = (j - 2) * LEN_2D + i
        prod = bb(idx) * cc(idx)
        aa(idx) = aa(prev_idx) + prod
      end do
    end if
  end do
  !$omp end parallel do

  return
end subroutine tsvc_2_s275_fp64
