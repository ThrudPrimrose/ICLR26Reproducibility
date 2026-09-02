subroutine ext_break_post_body_fp64(a, b, c, len_1d) bind(c, name='ext_break_post_body_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib, only: omp_get_thread_num, omp_get_num_threads
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*)
  real(c_double), intent(in)    :: c(*)
  integer(c_int64_t), value     :: len_1d

  integer(c_int64_t) :: n, i, cut
  integer(c_int64_t) :: lo, hi, bs, local_cut, j
  integer :: tid, nt

  n = len_1d
  if (n <= 0) return

  ! For small problems use the direct scalar loop to avoid OpenMP overhead.
  if (n < 4096_c_int64_t) then
    do i = 1, n
      a(i) = a(i) + b(i) * c(i)
      if (c(i) > b(i)) exit
    end do
    return
  end if

  cut = n + 1

  !$omp parallel default(none) &
  !$omp shared(a, b, c, n, cut) &
  !$omp private(tid, nt, bs, lo, hi, local_cut, j)
  tid = omp_get_thread_num()
  nt  = omp_get_num_threads()
  bs  = (n + int(nt, c_int64_t) - 1_c_int64_t) / int(nt, c_int64_t)
  lo  = int(tid, c_int64_t) * bs + 1_c_int64_t
  hi  = min(lo + bs - 1_c_int64_t, n)

  ! Find the first index in this thread's block where c(i) > b(i).
  local_cut = n + 1
  if (lo <= hi) then
    do j = lo, hi
      if (c(j) > b(j)) then
        local_cut = j
        exit
      end if
    end do
  end if

  !$omp critical
    if (local_cut < cut) cut = local_cut
  !$omp end critical

  !$omp barrier

  ! Update a(i) for all i up to and including the break index.
  if (cut <= n) then
    hi = min(hi, cut)
    if (lo <= hi) then
      !$omp simd
      do j = lo, hi
        a(j) = a(j) + b(j) * c(j)
      end do
      !$omp end simd
    end if
  end if
  !$omp end parallel
end subroutine ext_break_post_body_fp64
