subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: K, LEN_1D, workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int64_t) :: block_id, num_blocks, start_idx, end_idx, i

  if (LEN_1D <= K) return

  if (K == 1) then
    do i = 2, LEN_1D
      a(i) = 0.75d0 * a(i - 1) + b(i) * c(i)
    end do
  else
    num_blocks = (LEN_1D - 1) / K
    do block_id = 1, num_blocks
      start_idx = block_id * K + 1
      end_idx = min(start_idx + K - 1, LEN_1D)
      !$omp simd
      do i = start_idx, end_idx
        a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
      end do
      !$omp end simd
    end do
  end if
end subroutine versioned_distance_update_fp64
