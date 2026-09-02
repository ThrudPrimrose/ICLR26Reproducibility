subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t), parameter :: BS = 64
  integer(c_int64_t) :: m, nt, td, ti, tj, i, j, i0, i1, j0, j1, tjmin, tjmax

  m = LEN_2D - 1
  nt = (m + BS - 1) / BS

  !$omp parallel private(td, ti, tj, i, j, i0, i1, j0, j1, tjmin, tjmax)
  do td = 0, 2*nt - 2
    tjmin = max(0_c_int64_t, td - nt + 1)
    tjmax = min(nt - 1, td)
    !$omp do
    do tj = tjmin, tjmax
      ti = td - tj
      i0 = 2 + ti*BS
      i1 = min(LEN_2D, i0 + BS - 1)
      j0 = 2 + tj*BS
      j1 = min(LEN_2D, j0 + BS - 1)
      do i = i0, i1
        !$omp simd
        do j = j0, j1
          aa(j, i) = aa(j - 1, i - 1) + bb(j, i)
        end do
      end do
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s119_fp64
