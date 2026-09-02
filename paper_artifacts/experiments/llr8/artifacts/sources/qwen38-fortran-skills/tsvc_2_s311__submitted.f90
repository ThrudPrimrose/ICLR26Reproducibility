pure function chunk_sum(x, lo, hi) result(r)
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: x(*)
  integer(c_int64_t), value, intent(in) :: lo, hi
  real(c_double) :: r
  integer(c_int64_t) :: i
  r = 0.0d0
  !$omp simd reduction(+:r)
  do i = lo, hi
    r = r + x(i)
  end do
end function chunk_sum

subroutine tsvc_2_s311_fp64(a, sum_out, len_1d, ws, ws_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, ws_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(inout) :: sum_out(len_1d)
  integer(c_int8_t), intent(inout) :: ws(ws_size)
  real(c_double) :: s, part(64)
  interface
    pure function chunk_sum(x, lo, hi) result(r)
      use iso_c_binding
      real(c_double), intent(in) :: x(*)
      integer(c_int64_t), value, intent(in) :: lo, hi
      real(c_double) :: r
    end function chunk_sum
  end interface
  integer(c_int64_t) :: i, k, n
  integer :: nt

  n = len_1d
  if (n <= 0) return

  nt = omp_get_max_threads()
  if (nt > 64) nt = 64
  if (nt < 1) nt = 1

  if (n < 131072) then
    s = 0.0d0
    !$omp simd reduction(+:s)
    do i = 1, n
      s = s + a(i)
    end do
    sum_out(1) = s
    return
  end if

  ! Stage 1: each chunk is summed by a fixed-order SIMD reduction over a
  ! disjoint range; the stage is parallel across chunks. Stage 2 combines the
  ! chunk sums in a fixed sequential order, so the result is bit-identical
  ! across runs and across thread counts.
  !$omp parallel do schedule(static)
  do k = 1, nt
    part(k) = chunk_sum(a, (n * (k - 1)) / nt + 1, (n * k) / nt)
  end do

  s = 0.0d0
  do k = 1, nt
    s = s + part(k)
  end do
  sum_out(1) = s
end subroutine tsvc_2_s311_fp64

