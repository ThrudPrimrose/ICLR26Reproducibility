subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t), parameter :: B = 32
  integer(c_int64_t) :: ti, tj, td, ntiles_i, ntiles_j
  integer(c_int64_t) :: i0, i1, j0, j1, i, j

  ntiles_i = (LEN_2D + B - 1) / B
  ntiles_j = (LEN_2D + B - 1) / B

  !$omp parallel
  do td = 0, ntiles_i + ntiles_j - 2
    !$omp do schedule(static)
    do ti = max(0_c_int64_t, td - ntiles_j + 1), min(td, ntiles_i - 1)
      tj = td - ti
      i0 = ti * B + 1
      i1 = min((ti + 1) * B, LEN_2D)
      j0 = tj * B + 1
      j1 = min((tj + 1) * B, LEN_2D)
      do i = max(i0, 2_c_int64_t), i1
        !$omp simd
        do j = max(j0, 2_c_int64_t), j1
          aa(j, i) = aa(j - 1, i - 1) + bb(j, i)
        end do
      end do
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s119_fp64
