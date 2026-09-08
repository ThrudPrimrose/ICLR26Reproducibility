module tsvc_2_s2275
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(c)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D), aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: b(LEN_2D), bb(LEN_2D, LEN_2D), c(LEN_2D), cc(LEN_2D, LEN_2D), d(LEN_2D)
    integer(c_int64_t) :: i, j
    !$omp parallel private(i, j)
    !$omp do
    do j = 1, LEN_2D
      do i = 1, LEN_2D
        aa(i, j) = aa(i, j) + bb(i, j) * cc(i, j)
      end do
    end do
    !$omp end do nowait
    !$omp do
    do i = 1, LEN_2D
      a(i) = b(i) + c(i) * d(i)
    end do
    !$omp end do
    !$omp end parallel
  end subroutine tsvc_2_s2275_fp64
end module tsvc_2_s2275
