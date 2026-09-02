subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(c, name='tsvc_2_s235_fp64')
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a
  real(c_double), dimension(*), intent(inout) :: aa
  real(c_double), dimension(*), intent(in)    :: b
  real(c_double), dimension(*), intent(in)    :: bb
  real(c_double), dimension(*), intent(in)    :: c
  integer(c_int64_t), value, intent(in) :: len_2d

  integer(c_int64_t) :: n, ic, j, nfull, i
  real(c_double) :: acc(1:8), av(1:8)

  n = len_2d
  if (n <= 0) return

  nfull = (n - 1) / 8          ! number of full 8-column blocks

  !$omp parallel do default(none) shared(a, b, c, aa, bb, n, nfull) private(ic, j, acc, av)
  do ic = 1, 8 * nfull, 8
     av = a(ic:ic + 7) + b(ic:ic + 7) * c(ic:ic + 7)
     a(ic:ic + 7) = av
     acc = aa(ic:ic + 7)
     do j = 2, n
        acc = acc + bb((j - 1) * n + ic:(j - 1) * n + ic + 7) * av
        aa((j - 1) * n + ic:(j - 1) * n + ic + 7) = acc
     end do
  end do
  !$omp end parallel do

  ! tail columns (fewer than 8)
  do i = 8 * nfull + 1, n
     a(i) = a(i) + b(i) * c(i)
     do j = 2, n
        aa((j - 1) * n + i) = aa((j - 2) * n + i) + bb((j - 1) * n + i) * a(i)
     end do
  end do
end subroutine tsvc_2_s235_fp64
