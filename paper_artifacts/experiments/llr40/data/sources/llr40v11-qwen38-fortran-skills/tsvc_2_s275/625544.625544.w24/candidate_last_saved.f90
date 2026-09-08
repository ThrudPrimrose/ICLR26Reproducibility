subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j
  real(c_double) :: s

  !$omp parallel do schedule(static)
  do i = 1, len_2d
    if (aa(i, 1) > 0.0d0) then
      s = aa(i, 1)
      do j = 2, len_2d
        s = s + bb(i, j) * cc(i, j)
        aa(i, j) = s
      end do
    end if
  end do
  !$omp end parallel do
end subroutine tsvc_2_s275_fp64
