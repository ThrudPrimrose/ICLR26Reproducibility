module tsvc_2_s2275_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name="tsvc_2_s2275_fp64")
    real(c_double), intent(inout) :: a(0:*)
    real(c_double), intent(inout) :: aa(0:*)
    real(c_double), intent(in) :: b(0:*), bb(0:*), c(0:*), cc(0:*), d(0:*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j, idx
    !$omp parallel do default(none) shared(a, aa, b, bb, c, cc, d, LEN_2D) private(i, j, idx)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      !$omp simd
      do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
        idx = j * LEN_2D + i
        aa(idx) = aa(idx) + bb(idx) * cc(idx)
      end do
      a(i) = b(i) + c(i) * d(i)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s2275_fp64
end module tsvc_2_s2275_mod
