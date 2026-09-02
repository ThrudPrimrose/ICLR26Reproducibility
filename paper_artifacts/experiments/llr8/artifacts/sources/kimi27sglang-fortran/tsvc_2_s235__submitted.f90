module tsvc_2_s235
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer, parameter :: BS = 256
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(c)
    real(c_double), dimension(*), intent(inout) :: a, aa
    real(c_double), dimension(*), intent(in) :: b, bb, c
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j, n, ib, ie

    n = LEN_2D

    !$omp parallel do default(none) private(i, j, ib, ie) shared(a, b, c, aa, bb, n) schedule(static)
    do ib = 1, n, BS
       ie = min(ib + BS - 1, n)
       do i = ib, ie
          a(i) = a(i) + b(i) * c(i)
       end do
       do j = 2, n
          !$omp simd
          do i = ib, ie
             aa((j - 1) * n + i) = aa((j - 2) * n + i) + bb((j - 1) * n + i) * a(i)
          end do
          !$omp end simd
       end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235
