subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(C, name='tsvc_2_s1232_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_2d
  integer(c_int64_t), value :: vlen
  real(c_double), intent(inout), dimension(len_2d*len_2d) :: aa
  real(c_double), intent(in),    dimension(len_2d*len_2d) :: bb
  real(c_double), intent(in),    dimension(len_2d*len_2d) :: cc

  integer(c_int64_t) :: n, i, base, jmax
  integer(c_int32_t) :: j
  n = len_2d

  if (vlen <= 0) then
    !$omp parallel do schedule(static)
    do i = 0, n - 1
      base = i * n
      !$omp simd
      do j = 0, int(n, 4) - 1
        aa(base + 1 + j) = bb(base + 1 + j) + cc(base + 1 + j)
      end do
    end do
  else
    !$omp parallel do schedule(static)
    do i = 0, n - 1
      jmax = i / vlen
      if (jmax > n - 1) jmax = n - 1
      base = i * n
      !$omp simd
      do j = 0, int(jmax, 4)
        aa(base + 1 + j) = bb(base + 1 + j) + cc(base + 1 + j)
      end do
    end do
  end if
end subroutine tsvc_2_s1232_fp64
