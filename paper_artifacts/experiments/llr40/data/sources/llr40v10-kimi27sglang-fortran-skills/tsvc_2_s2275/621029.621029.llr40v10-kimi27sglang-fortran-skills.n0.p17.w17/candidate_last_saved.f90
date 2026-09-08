subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  real(c_double), intent(inout) :: a(LEN_2D), aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D), bb(LEN_2D, LEN_2D), c(LEN_2D), cc(LEN_2D, LEN_2D), d(LEN_2D)
  integer :: i, j, n

  n = int(LEN_2D)

  !$omp parallel
    !$omp do simd
    do i = 1, n
      a(i) = b(i) + c(i) * d(i)
    end do
    !$omp end do simd

    !$omp do schedule(static)
    do j = 1, n
      !$omp simd
      do i = 1, n
        aa(i, j) = aa(i, j) + bb(i, j) * cc(i, j)
      end do
      !$omp end simd
    end do
    !$omp end do
  !$omp end parallel
end subroutine tsvc_2_s2275_fp64
