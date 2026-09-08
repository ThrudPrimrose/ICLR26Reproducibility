subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D)
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: c(LEN_2D)
  integer(c_int64_t) :: i, j, jm

  !$omp parallel private(i, j, jm)
  !$omp do simd schedule(static) nowait
  do i = 1, LEN_2D
    a(i) = a(i) + b(i) * c(i)
  end do

  do j = 2, LEN_2D - 1, 2
    jm = j - 1
    !$omp do simd schedule(static) nowait
    do i = 1, LEN_2D
      aa(i, j) = aa(i, jm) + bb(i, j) * a(i)
      aa(i, j + 1) = aa(i, jm) + (bb(i, j) + bb(i, j + 1)) * a(i)
    end do
  end do

  if (mod(LEN_2D, 2_c_int64_t) == 0_c_int64_t) then
    j = LEN_2D
    !$omp do simd schedule(static) nowait
    do i = 1, LEN_2D
      aa(i, j) = aa(i, j - 1) + bb(i, j) * a(i)
    end do
  end if
  !$omp end parallel
end subroutine tsvc_2_s235_fp64
