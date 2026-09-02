subroutine tsvc_2_s119_fp64(aa, bb, len_2d, ws, ws_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d, ws_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(inout) :: ws(0:ws_size - 1)
  integer(c_int64_t) :: i, j
  if (ws_size > 0) ws(0) = 0.0d0
  do i = 2, len_2d
    !$omp parallel do simd
    do j = 2, len_2d
      aa(j, i) = aa(j - 1, i - 1) + bb(j, i)
    end do
  end do
end subroutine tsvc_2_s119_fp64
