subroutine tsvc_2_s275_fp64(aa, bb, cc, n, workspace, workspace_size) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  real(c_double), intent(inout) :: aa(n, n)
  real(c_double), intent(in) :: bb(n, n), cc(n, n)
  integer(c_int8_t), intent(in) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j

  !$omp parallel private(i, j)
  do j = 2, n
     !$omp do simd
     do i = 1, n
        if (aa(i, 1) > 0.0d0) then
           aa(i, j) = aa(i, j - 1) + bb(i, j) * cc(i, j)
        end if
     end do
     !$omp end do simd
  end do
  !$omp end parallel
end subroutine tsvc_2_s275_fp64
