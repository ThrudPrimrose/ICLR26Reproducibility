subroutine wf_diff_skew_fp64(a, n) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n, n)
  integer(c_int64_t) :: i, j

  !$omp parallel private(i, j)
  do i = 2, n
    !$omp do simd
    do j = 1, n - 1
      a(j, i) = a(j, i) + a(j, i - 1) + a(j + 1, i - 1)
    end do
    !$omp end do simd
  end do
  !$omp end parallel
end subroutine wf_diff_skew_fp64
