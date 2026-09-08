module tsvc_2_s235
  use iso_c_binding, only: c_int64_t, c_double
  use iso_fortran_env, only: int64
  implicit none
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(c, name='tsvc_2_s235_fp64')
    integer(c_int64_t), intent(in), value :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D), aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D), bb(LEN_2D, LEN_2D), c(LEN_2D)
    integer(int64) :: i, j

    !$omp simd
    do i = 1, LEN_2D
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end simd

    !$omp parallel private(j)
    do j = 2, LEN_2D
      !$omp do simd
      do i = 1, LEN_2D
        aa(i, j) = aa(i, j-1) + bb(i, j) * a(i)
      end do
      !$omp end do simd
    end do
    !$omp end parallel
  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235
