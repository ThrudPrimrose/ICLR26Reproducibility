subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(1)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t), parameter :: nchunk = 1536
  integer(c_int64_t) :: m, k, c, lo, hi, q, t
  real(c_double) :: s, s2, s21, s22, s23, s24, s25, s26, s27, s28
  real(c_double), allocatable :: part(:)

  m = len_1d / 2
  s = 0.0d0
  if (m >= 131072_c_int64_t) then
    allocate(part(nchunk))
    !$omp parallel do private(lo, hi, q, t, s2, s21, s22, s23, s24, s25, s26, s27, s28)
    do c = 1, nchunk
      lo = (m * (c - 1)) / nchunk + 1
      hi = (m * c) / nchunk
      q = (hi - lo + 1) / 8
      s21 = 0.0d0
      s22 = 0.0d0
      s23 = 0.0d0
      s24 = 0.0d0
      s25 = 0.0d0
      s26 = 0.0d0
      s27 = 0.0d0
      s28 = 0.0d0
      do k = 1, q
        s21 = s21 + a(2*(lo + k - 1))
        s22 = s22 + a(2*(lo + q + k - 1))
        s23 = s23 + a(2*(lo + 2*q + k - 1))
        s24 = s24 + a(2*(lo + 3*q + k - 1))
        s25 = s25 + a(2*(lo + 4*q + k - 1))
        s26 = s26 + a(2*(lo + 5*q + k - 1))
        s27 = s27 + a(2*(lo + 6*q + k - 1))
        s28 = s28 + a(2*(lo + 7*q + k - 1))
      end do
      s2 = (((((s21 + s22) + s23) + s24) + s25) + s26) + s27 + s28
      do t = lo + 8*q, hi
        s2 = s2 + a(2*t)
      end do
      part(c) = s2
    end do
    do c = 1, nchunk
      s = s + part(c)
    end do
  else
    !$omp simd reduction(+:s)
    do k = 1, m
      s = s + a(2*k)
    end do
    !$omp end simd
  end if
  out(1) = s
  if (workspace_size > 0) workspace(1) = workspace(1)
end subroutine quasi_affine_reduce_odd_fp64
