subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(C, name="tsvc_2_s235_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)
  integer(c_int64_t) :: n, nb, ib, i0, m, j, k
  real(c_double) :: acc(32), av(32)
  real(c_double) :: ws(32)

  n = len_2d
  if (n <= 0) return
  nb = (n + 31) / 32

  if (n >= 512) then
    !$omp parallel do schedule(dynamic, 1) &
    !$omp& default(none) shared(a, aa, bb, b, c, n, nb) &
    !$omp& private(ib, i0, m, j, k, acc, av, ws)
    do ib = 1, nb
      i0 = (ib - 1) * 32 + 1
      m = min(32_8, n - i0 + 1)
      do k = 1, m
        ws(k) = a(i0 + k - 1) + b(i0 + k - 1) * c(i0 + k - 1)
        a(i0 + k - 1) = ws(k)
        av(k) = ws(k)
        acc(k) = aa(i0 + k - 1, 1)
      end do
      do j = 2, n
        do k = 1, m
          acc(k) = acc(k) + bb(i0 + k - 1, j) * av(k)
        end do
        do k = 1, m
          aa(i0 + k - 1, j) = acc(k)
        end do
      end do
    end do
  else
    do ib = 1, nb
      i0 = (ib - 1) * 32 + 1
      m = min(32_8, n - i0 + 1)
      do k = 1, m
        ws(k) = a(i0 + k - 1) + b(i0 + k - 1) * c(i0 + k - 1)
        a(i0 + k - 1) = ws(k)
        av(k) = ws(k)
        acc(k) = aa(i0 + k - 1, 1)
      end do
      do j = 2, n
        do k = 1, m
          acc(k) = acc(k) + bb(i0 + k - 1, j) * av(k)
        end do
        do k = 1, m
          aa(i0 + k - 1, j) = acc(k)
        end do
      end do
    end do
  end if
end subroutine tsvc_2_s235_fp64
