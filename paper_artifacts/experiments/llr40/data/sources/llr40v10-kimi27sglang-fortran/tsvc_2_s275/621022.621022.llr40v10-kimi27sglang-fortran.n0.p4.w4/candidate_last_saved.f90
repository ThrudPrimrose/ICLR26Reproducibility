module tsvc_2_s275_mod
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D, workspace, workspace_bytes) bind(c)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: aa(0:*)
    real(c_double), intent(in)    :: bb(0:*)
    real(c_double), intent(in)    :: cc(0:*)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_bytes
    integer(c_int64_t) :: i, j, n, idx, idxm
    n = LEN_2D
    !$omp parallel do schedule(static) private(i,j,idx,idxm)
    do i = 0, n-1
      if (aa(i) > 0.0_c_double) then
        do j = 1, n-1
          idx  = j*n + i
          idxm = idx - n
          aa(idx) = aa(idxm) + bb(idx) * cc(idx)
        end do
      end if
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
