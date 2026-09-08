module wf_diff_skew_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
    ! Arguments: a is a double-precision array (flattened 2D), LEN_2D is its dimension (square matrix)
    real(c_double), dimension(*), intent(inout) :: a
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx, idx_up, idx_up_right
    !$omp parallel default(shared) private(i, j, idx, idx_up, idx_up_right)
    do i = 1_c_int64_t, LEN_2D-1_c_int64_t
      !$omp do simd schedule(static)
      do j = 0_c_int64_t, LEN_2D-2_c_int64_t
        idx = i * LEN_2D + j
        idx_up = (i - 1_c_int64_t) * LEN_2D + j
        idx_up_right = (i - 1_c_int64_t) * LEN_2D + (j + 1_c_int64_t)
        a(idx + 1) = a(idx + 1) + a(idx_up + 1) + a(idx_up_right + 1)
      end do
      !$omp end do simd
    end do
    !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
