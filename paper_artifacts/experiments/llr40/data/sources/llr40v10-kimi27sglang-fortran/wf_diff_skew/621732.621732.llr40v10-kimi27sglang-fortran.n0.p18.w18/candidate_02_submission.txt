subroutine wf_diff_skew_fp64(a, LEN_2D) bind(c, name="wf_diff_skew_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: a(0_c_int64_t : LEN_2D * LEN_2D - 1_c_int64_t)
  integer(c_int64_t) :: i, j, n

  n = LEN_2D
  if (n <= 1_c_int64_t) return

  do i = 1_c_int64_t, n - 1_c_int64_t
    !$omp simd
    do j = 0_c_int64_t, n - 2_c_int64_t
      a(i * n + j) = a(i * n + j) + a((i - 1_c_int64_t) * n + j) &
                                    + a((i - 1_c_int64_t) * n + j + 1_c_int64_t)
    end do
  end do
end subroutine wf_diff_skew_fp64
