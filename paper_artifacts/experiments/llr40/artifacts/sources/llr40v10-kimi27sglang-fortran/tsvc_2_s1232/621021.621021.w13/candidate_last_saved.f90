subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(c, name="tsvc_2_s1232_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN
  real(c_double), intent(inout) :: aa(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: bb(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: cc(0:LEN_2D*LEN_2D-1)
  integer(c_int64_t) :: i, j, ij, jmax

  !$omp parallel do schedule(dynamic, 8) private(j, ij, jmax)
  do i = 0, LEN_2D - 1
    jmax = min(LEN_2D - 1, i / VLEN)
    do j = 0, jmax
      ij = i * LEN_2D + j
      aa(ij) = bb(ij) + cc(ij)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s1232_fp64
