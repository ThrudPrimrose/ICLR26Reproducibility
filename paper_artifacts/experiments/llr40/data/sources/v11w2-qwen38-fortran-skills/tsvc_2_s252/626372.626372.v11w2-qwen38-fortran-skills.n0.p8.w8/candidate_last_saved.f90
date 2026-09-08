subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(out) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  integer(c_int64_t) :: i, k, nb
  real(c_double) :: s1, s2, s3, s4, s5, s6

  if (len_1d < 1) return
  a(1) = b(1) * c(1)
  nb = (len_1d - 1) / 8
  !$omp parallel do simd
  do k = 0, nb - 1
    i = 2 + 8 * k
    s1 = b(i) * c(i)
    s2 = b(i + 1) * c(i + 1)
    s3 = b(i + 2) * c(i + 2)
    s4 = b(i + 3) * c(i + 3)
    s5 = b(i + 4) * c(i + 4)
    s6 = b(i + 5) * c(i + 5)
    a(i) = s1 + b(i - 1) * c(i - 1)
    a(i + 1) = s2 + s1
    a(i + 2) = s3 + s2
    a(i + 3) = s4 + s3
    a(i + 4) = s5 + s4
    a(i + 5) = s6 + s5
    a(i + 6) = b(i + 6) * c(i + 6) + s6
    a(i + 7) = b(i + 7) * c(i + 7) + b(i + 6) * c(i + 6)
  end do
  do i = 2 + 8 * nb, len_1d
    a(i) = b(i) * c(i) + b(i - 1) * c(i - 1)
  end do
end subroutine tsvc_2_s252_fp64
