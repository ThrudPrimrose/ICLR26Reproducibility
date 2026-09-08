subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d), b(len_1d)
  real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)

  real(c_double) :: t
  integer(c_int64_t) :: i

  ! Fused elementwise kernel (both reference loops touch index i only):
  !   b(i) = d(i) * e(i);  a(i) = a(i) + b(i) * c(i)
  ! Every iteration is independent and unit stride: threads + SIMD.
  ! Compute through the temporary so the store to b is not re-loaded.
  !$omp parallel do simd
  do i = 1, len_1d
    t = d(i) * e(i)
    b(i) = t
    a(i) = a(i) + t * c(i)
  end do
end subroutine tsvc_2_s152_fp64
