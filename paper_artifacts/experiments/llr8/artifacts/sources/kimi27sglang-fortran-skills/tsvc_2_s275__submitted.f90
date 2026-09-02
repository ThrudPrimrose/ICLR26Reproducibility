subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d), cc(len_2d, len_2d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j, jj, j_end
  integer(c_int64_t), parameter :: tile = 16

  !$omp parallel private(i, j, jj, j_end)
  do jj = 2, len_2d, tile
    j_end = min(len_2d, jj + tile - 1)
    do j = jj, j_end - 1
      !$omp do simd schedule(static) nowait
      do i = 1, len_2d
        if (aa(i, 1) > 0.0d0) then
          aa(i, j) = aa(i, j - 1) + bb(i, j) * cc(i, j)
        end if
      end do
      !$omp end do simd
    end do
    !$omp do simd schedule(static)
    do i = 1, len_2d
      if (aa(i, 1) > 0.0d0) then
        aa(i, j_end) = aa(i, j_end - 1) + bb(i, j_end) * cc(i, j_end)
      end if
    end do
    !$omp end do simd
  end do
  !$omp end parallel
end subroutine tsvc_2_s275_fp64
