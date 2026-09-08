module tsvc_2_s235_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D)
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: c(LEN_2D)
    integer(c_int64_t) :: i, j
        a = a + b * c
        !$omp parallel private(i)
        do j = 2, LEN_2D
          !$omp do simd schedule(static) nowait
          do i = 1, LEN_2D
            aa(i, j) = aa(i, j-1) + bb(i, j) * a(i)
          end do
          !$omp end do simd
        end do
        !$omp end parallel
  end subroutine tsvc_2_s235_fp64

end module tsvc_2_s235_mod
