module wf_diff_skew_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: a(0:*)
    integer(c_int64_t) :: i, j, base_i, base_im1
    !$omp parallel default(none) shared(a, LEN_2D) private(i, j, base_i, base_im1)
    do i = 1, LEN_2D - 1
            base_i = i * LEN_2D
      base_im1 = (i - 1) * LEN_2D
      !$omp do simd schedule(static)
      do j = 0, LEN_2D - 2
        a(base_i + j) = a(base_i + j) + a(base_im1 + j) + a(base_im1 + j + 1)
      end do
      !$omp end do simd
    end do
    !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
