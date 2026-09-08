module tsvc_2_s323_mod
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
    real(c_double), dimension(0:*), intent(inout) :: a, b
    real(c_double), dimension(0:*), intent(in) :: c, d, e
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, n
    real(c_double) :: acc
    n = LEN_1D
    if (n <= 1_c_int64_t) return

    ! Precompute products into output arrays (a <- c*d, b <- c*e) vectorised.
    !$omp parallel do simd
    do i = 1_c_int64_t, n - 1_c_int64_t
      a(i) = c(i) * d(i)
      b(i) = c(i) * e(i)
    end do
    !$omp end parallel do simd

    ! Sequential recurrence over the precomputed products.
    acc = b(0_c_int64_t)
    do i = 1_c_int64_t, n - 1_c_int64_t
      a(i) = acc + a(i)
      acc = a(i) + b(i)
      b(i) = acc
    end do
  end subroutine
end module
