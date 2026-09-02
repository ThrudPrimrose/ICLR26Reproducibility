module tsvc_2_s235_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: bb(*)
    real(c_double), intent(in) :: c(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    ! Update vector a in parallel
    !$omp parallel
!$omp do schedule(static)
    do i = 1, LEN_2D
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end do
!$omp end parallel
    ! Update matrix aa row-wise (flattened row-major layout)
    do j = 2, LEN_2D
      !$omp simd
      do i = 1, LEN_2D
        aa((j-1)*LEN_2D + i) = aa((j-2)*LEN_2D + i) + bb((j-1)*LEN_2D + i) * a(i)
      end do
    end do
  end subroutine tsvc_2_s235_fp64
end module tsvc_2_s235_mod
