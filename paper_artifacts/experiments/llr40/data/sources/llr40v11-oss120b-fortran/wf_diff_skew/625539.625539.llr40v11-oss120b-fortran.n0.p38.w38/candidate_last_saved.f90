module wf_diff_skew_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
    real(c_double), intent(inout) :: a(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: base_i, base_up
    !$omp parallel private(i, j, base_i, base_up)
    do i = 1_c_int64_t, LEN_2D - 1_c_int64_t
      base_i = i * LEN_2D
      base_up = (i - 1_c_int64_t) * LEN_2D
      !$omp do
      do j = 0_c_int64_t, LEN_2D - 2_c_int64_t
        a(base_i + j + 1_c_int64_t) = a(base_i + j + 1_c_int64_t) + a(base_up + j + 1_c_int64_t) + a(base_up + j + 2_c_int64_t)
      end do
      !$omp end do
    end do
    !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
