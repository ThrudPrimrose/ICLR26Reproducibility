subroutine tsvc_2_s3110_fp64_impl(aa, bb, len_2d) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(in) :: aa(len_2d * len_2d)
  real(c_double), intent(inout) :: bb(4)

  integer(c_int64_t) :: n, i, first, xindex, yindex
  real(c_double) :: m, chksum

  n = len_2d * len_2d
  if (n == 0) return

  ! Pass 1: maximum value (reduction is exact: max picks a stored element)
  m = -HUGE(0.0d0)
  !$omp parallel do simd reduction(max:m)
  do i = 1, n
    m = max(m, aa(i))
  end do

  ! Pass 2: FIRST occurrence of the maximum (row-major flat order),
  ! matches the reference's strict '>' first-hit tie-break.
  first = n + 1
  !$omp parallel do simd reduction(min:first)
  do i = 1, n
    if (aa(i) == m) first = min(first, i)
  end do

  xindex = (first - 1) / len_2d
  yindex = (first - 1) - xindex * len_2d
  chksum = m + dble(xindex) + dble(yindex)
  bb(1) = chksum
end subroutine tsvc_2_s3110_fp64_impl
