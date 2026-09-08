module scan_affine_decay_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
contains
  subroutine step(i, y, c, x)
    integer(c_int64_t), intent(in) :: i
    real(c_double), intent(inout) :: y(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    y(i) = c(i) * y(i-1) + x(i)
  end subroutine step

  subroutine scan_affine_decay_fp64(y, c, x, LEN_1D) bind(C, name="scan_affine_decay_fp64")
    real(c_double), intent(inout) :: y(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, n
    if (LEN_1D <= 0) return
    n = LEN_1D
    y(1) = x(1)
    if (n <= 1) return
    !$omp parallel default(none) shared(y,c,x,n) private(i)
    !$omp single
    i = 2_c_int64_t
    do while (i <= n)
      call step(i, y, c, x)
      i = i + 1_c_int64_t
    end do
    !$omp end single
    !$omp end parallel
  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
