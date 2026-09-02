subroutine wf_diff_skew_fp64(a, LEN_2D) bind(c, name='wf_diff_skew_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j
  do i = 2, LEN_2D
    !$omp simd safelen(64)
    do j = 1, LEN_2D - 1
      a(j, i) = a(j, i) + a(j, i - 1) + a(j + 1, i - 1)
    end do
    !$omp end simd
  end do
end subroutine wf_diff_skew_fp64
