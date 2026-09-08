subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  !$omp parallel do schedule(static) private(j)
  do i = 1, LEN_2D
    if (aa(i, 1) > 0.0_c_double) then
      do j = 2, LEN_2D
        aa(i, j) = aa(i, j-1) + bb(i, j) * cc(i, j)
      end do
    end if
  end do
  !$omp end parallel do

end subroutine tsvc_2_s275_fp64
