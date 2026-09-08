subroutine ext_war_unit_fp64(a, b, len1d) bind(C, name='ext_war_unit_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*)
  integer(c_int64_t), intent(in), value :: len1d

  integer(c_int64_t) :: n, i, k, T, s, e, chunk
  real(c_double), allocatable :: boundary(:)

  n = len1d
  if (n < 2_8) return
  if (n == 2_8) then
    a(1) = a(2) + b(1)
    return
  end if

  T = 24_8
  call omp_set_num_threads(int(T))
  allocate(boundary(T))
  chunk = (n - 1_8 + T - 1_8) / T
  do k = 1_8, T
    e = k * chunk
    if (e > n - 1_8) e = n - 1_8
    boundary(k) = a(e + 1_8)
  end do
  !$omp parallel do schedule(static)
  do k = 1_8, T
    s = (k - 1_8) * chunk + 1_8
    e = k * chunk
    if (e > n - 1_8) e = n - 1_8
    if (s > e) cycle
    do i = s, e - 1_8
      a(i) = a(i + 1_8) + b(i)
    end do
    a(e) = boundary(k) + b(e)
  end do
  !$omp end parallel do
end subroutine
