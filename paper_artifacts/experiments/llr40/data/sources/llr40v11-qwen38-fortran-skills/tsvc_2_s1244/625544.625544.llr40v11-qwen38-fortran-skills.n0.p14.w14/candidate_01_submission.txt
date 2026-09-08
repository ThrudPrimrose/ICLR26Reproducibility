subroutine tsvc_2_s1244_fp64(a, b, c, d, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(inout) :: d(len_1d)

  integer(c_int64_t) :: n, i
  real(c_double) :: t

  n = len_1d
  if (n < 2) return

  ! Pass 1: stash the original a(i+1) values into d (i = 1..n-1).
  ! a(n) is never written, and a(i+1) is read here BEFORE any write to it.
  !$omp parallel do simd schedule(static)
  do i = 1, n - 1
    d(i) = a(i + 1)
  end do

  ! Pass 2: compute the new a(i) and finish d(i) = t(i) + a_orig(i+1).
  ! Fully independent: a(i), d(i) are written at their own index only.
  !$omp parallel do simd schedule(static)
  do i = 1, n - 1
    t = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    a(i) = t
    d(i) = t + d(i)
  end do
end subroutine tsvc_2_s1244_fp64
