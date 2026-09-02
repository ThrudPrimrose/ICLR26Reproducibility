subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s231_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), dimension(LEN_2D, LEN_2D), intent(inout) :: aa
  real(c_double), dimension(LEN_2D, LEN_2D), intent(in) :: bb
  integer(c_int64_t) :: i, j, ib, iend
  integer(c_int64_t), parameter :: BLOCK = 512
  !$omp parallel do schedule(static) private(j, i, iend)
  do ib = 1, LEN_2D, BLOCK
    iend = min(ib + BLOCK - 1, LEN_2D)
    do j = 2, LEN_2D
      !$omp simd
      do i = ib, iend
        aa(i, j) = aa(i, j-1) + bb(i, j)
      end do
    end do
  end do
end subroutine tsvc_2_s231_fp64
