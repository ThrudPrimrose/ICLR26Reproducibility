subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  !$omp parallel default(none) shared(a, LEN_2D) private(i, j)
  do i = 2_c_int64_t, LEN_2D
    !$omp do schedule(static)
    do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
      a(j, i) = a(j, i) + a(j, i-1) + a(j+1, i-1)
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wf_diff_skew_fp64
