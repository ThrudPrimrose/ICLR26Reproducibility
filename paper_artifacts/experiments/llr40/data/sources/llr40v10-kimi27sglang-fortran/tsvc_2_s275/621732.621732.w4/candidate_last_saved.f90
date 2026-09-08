subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), dimension(LEN_2D, LEN_2D), intent(inout) :: aa
  real(c_double), dimension(LEN_2D, LEN_2D), intent(in) :: bb, cc
  integer(c_int64_t) :: i, j

  !$omp parallel do simd private(j) schedule(static)
  do i = 1_c_int64_t, LEN_2D
    if (aa(i, 1) > 0.0_c_double) then
      do j = 2_c_int64_t, LEN_2D
        aa(i, j) = aa(i, j - 1_c_int64_t) + bb(i, j) * cc(i, j)
      end do
    end if
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s275_fp64
