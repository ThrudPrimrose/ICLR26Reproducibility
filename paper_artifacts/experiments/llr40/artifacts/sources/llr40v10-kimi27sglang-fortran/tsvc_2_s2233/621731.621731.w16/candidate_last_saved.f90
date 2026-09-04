subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(c)
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D,LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D,LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D,LEN_2D)
  integer(c_int64_t) :: i, j

  do j = 9, LEN_2D
    !$omp simd
    do i = 9, LEN_2D
      aa(i,j) = aa(i,j-1) + cc(i,j)
    end do
  end do

  do i = 9, LEN_2D
    !$omp simd
    do j = 9, LEN_2D
      bb(j,i) = bb(j,i-1) + cc(j,i)
    end do
  end do
end subroutine
