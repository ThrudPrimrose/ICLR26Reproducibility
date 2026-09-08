subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d), cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j, i0, i_max
  integer(c_int64_t), parameter :: B = 512
  logical :: active(B)

  !$omp parallel do schedule(static) private(i0, i_max, j, i, active)
  do i0 = 1, len_2d, B
    i_max = min(i0 + B - 1, len_2d)
    do i = i0, i_max
      active(i - i0 + 1) = aa(i, 1) > 0.0d0
    end do
    do j = 2, len_2d
      !$omp simd
      do i = i0, i_max
        aa(i, j) = merge(aa(i, j - 1) + bb(i, j) * cc(i, j), aa(i, j), active(i - i0 + 1))
      end do
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s275_fp64
