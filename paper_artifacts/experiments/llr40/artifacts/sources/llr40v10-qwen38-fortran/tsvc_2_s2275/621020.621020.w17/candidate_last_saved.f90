subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C, name="tsvc_2_s2275_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(out)   :: a(*)
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)   :: b(*), bb(*), c(*), cc(*), d(*)
  integer(c_int64_t), value :: len_2d

  integer(c_int64_t) :: i, n, m

  n = len_2d
  m = n * n

  if (m <= 1048576) then
    do i = 1, m
      aa(i) = aa(i) + bb(i) * cc(i)
    end do
    do i = 1, n
      a(i) = b(i) + c(i) * d(i)
    end do
  else
    !$omp parallel default(none) shared(aa, bb, cc, a, b, c, d, m, n)
    !$omp do schedule(static)
    do i = 1, m
      aa(i) = aa(i) + bb(i) * cc(i)
    end do
    !$omp do schedule(static)
    do i = 1, n
      a(i) = b(i) + c(i) * d(i)
    end do
    !$omp end parallel
  end if
end subroutine tsvc_2_s2275_fp64
