subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_2D
  integer(c_int64_t), value :: VLEN
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in) :: bb(*)
  real(c_double), intent(in) :: cc(*)
  integer(c_int64_t) :: i, j, jmax, base, idx

  if (LEN_2D <= 256) then
    do i = 0, LEN_2D - 1
      base = i * LEN_2D
      jmax = i / VLEN
      do concurrent (j = 1:jmax + 1)
        idx = base + j
        aa(idx) = bb(idx) + cc(idx)
      end do
    end do
  else
    !$omp parallel do simd schedule(guided) private(j, jmax, base, idx)
    do i = 0, LEN_2D - 1
      base = i * LEN_2D
      jmax = i / VLEN
      do j = 1, jmax + 1
        idx = base + j
        aa(idx) = bb(idx) + cc(idx)
      end do
    end do
    !$omp end parallel do simd
  end if
end subroutine tsvc_2_s1232_fp64
