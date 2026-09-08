subroutine tsvc_2_s1244_fp64(a, b, c, d, len_1d) bind(C, name="tsvc_2_s1244_fp64")
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(inout) :: d(len_1d)
  integer(c_int64_t) :: n, i

  n = len_1d - 1
  if (n < 1) return

  ! a_new(i) = b(i) + c(i)^2 + b(i)^2 + c(i)
  ! d(i)     = a_new(i) + a_orig(i+1),  i = 1..len_1d-1
  ! The read a(i+1) is a future-element read (false dependence): rename it by
  ! staging the shifted original a into d, then finishing d in the 2nd pass.
  ! Both passes are independent across i -> parallel + simd.
  !$omp parallel do simd
  do i = 1, n
    d(i) = a(i + 1)
  end do
  !$omp parallel do simd
  do i = 1, n
    a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    d(i) = d(i) + a(i)
  end do
end subroutine tsvc_2_s1244_fp64
