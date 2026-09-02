subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name='tsvc_2_s235_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D)
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: c(LEN_2D)
  integer(c_int64_t) :: i, j, n

  n = LEN_2D

  !$omp parallel proc_bind(spread)

  !$omp do simd schedule(static)
  do i = 1, n
     a(i) = a(i) + b(i) * c(i)
  end do
  !$omp end do simd nowait

  do j = 2, n
     !$omp do simd schedule(static)
     do i = 1, n
        aa(i, j) = aa(i, j - 1) + bb(i, j) * a(i)
     end do
     !$omp end do simd nowait
  end do

  !$omp end parallel

end subroutine tsvc_2_s235_fp64
