module ext_break_find_first_m
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_find_first_fp64(a, b, c, d, LEN_1D) bind(c, name="ext_break_find_first_fp64")
    integer(c_int64_t), intent(in), value :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D), d(LEN_1D)
    integer(c_int64_t) :: i, first_neg, n
    n = LEN_1D
    first_neg = n + 1
    !$omp parallel if(n > 65536)
    !$omp do simd reduction(min:first_neg) schedule(static)
    do i = 1, n
      if (d(i) < 0.0_c_double) first_neg = min(first_neg, i)
    end do
    !$omp do simd schedule(static)
    do i = 1, first_neg - 1
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel
  end subroutine
end module
