! TSVC tsvc_2/s255:  a(i) = 0.333*(b(i) + b(i-1) + b(i-2))
! The x/y "recurrence" in the reference is a false dependence: after
! iteration i, x == b(i) and y == b(i-1) exactly.  Only the first two
! output elements need the wrapped values b(n-1), b(n-2).
subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(out) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)

  integer(c_int64_t) :: n, i
  real(c_double) :: c

  n = len_1d
  if (n < 2) then
     if (n == 1) a(1) = (b(1) + b(1) + b(1)) * 0.333d0
     return
  end if
  c = 0.333d0
  a(1) = (b(1) + b(n) + b(n - 1)) * c
  a(2) = (b(2) + b(1) + b(n)) * c
  if (n < 3) return
  !$omp parallel do simd
  do i = 3, n
     a(i) = (b(i) + b(i - 1) + b(i - 2)) * c
  end do
end subroutine tsvc_2_s255_fp64
