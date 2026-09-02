! Optimized Fortran implementation of argmax_with_index (TSVC s315).
! The judge's Fortran harness expects a 1-based index.
subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(c, name='argmax_with_index_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D

  integer(c_int64_t) :: n, i
  integer(c_int) :: nt, tid, j
  real(c_double) :: x
  integer(c_int64_t) :: idx
  integer(c_int64_t) :: start, finish, chunk, k

  integer(c_int), parameter :: VS = 8
  real(c_double) :: xv(VS)
  integer(c_int64_t) :: idxv(VS)

  n = LEN_1D
  if (n <= 0_c_int64_t) then
    out_value(1) = 0.0_c_double
    out_index(1) = 0_c_int64_t
    return
  end if
  if (n == 1_c_int64_t) then
    out_value(1) = a(1)
    out_index(1) = 1_c_int64_t
    return
  end if

  ! Small inputs: scalar scan beats parallel setup overhead.
  if (n < 4096_c_int64_t) then
    x = a(1)
    idx = 1_c_int64_t
    do i = 2_c_int64_t, n
      if (a(i) > x) then
        x = a(i)
        idx = i
      end if
    end do
    out_value(1) = x
    out_index(1) = idx
    return
  end if

  nt = omp_get_max_threads()
  block
    real(c_double) :: tval(nt)
    integer(c_int64_t) :: tidx(nt)

    !$omp parallel default(none) shared(a, n, tval, tidx, nt) &
    !$omp private(tid, start, finish, chunk, x, idx, k, j, xv, idxv)
    tid = omp_get_thread_num() + 1
    chunk = n / int(nt, c_int64_t)
    start = (int(tid, c_int64_t) - 1_c_int64_t) * chunk + 1_c_int64_t
    finish = int(tid, c_int64_t) * chunk
    if (tid == nt) finish = n

    ! Start from the global first element so that running-max semantics
    ! (e.g. NaN at position 1) are preserved by every thread.
    x = a(1)
    idx = 1_c_int64_t

    if (start <= finish) then
      ! Initialise per-lane candidates.
      do j = 1, VS
        xv(j) = a(start)
        idxv(j) = start
      end do

      ! Vector-lane scan over full VS-sized chunks.
      k = start
      do while (k + int(VS, c_int64_t) - 1_c_int64_t <= finish)
        do concurrent (j = 1:VS)
          if (a(k + int(j, c_int64_t) - 1_c_int64_t) > xv(j)) then
            xv(j) = a(k + int(j, c_int64_t) - 1_c_int64_t)
            idxv(j) = k + int(j, c_int64_t) - 1_c_int64_t
          end if
        end do
        k = k + int(VS, c_int64_t)
      end do

      ! Reduce lanes to the thread's scalar candidate.
      do j = 1, VS
        if (xv(j) > x) then
          x = xv(j)
          idx = idxv(j)
        end if
      end do

      ! Tail elements (fewer than VS).
      do i = k, finish
        if (a(i) > x) then
          x = a(i)
          idx = i
        end if
      end do
    end if

    tval(tid) = x
    tidx(tid) = idx
    !$omp end parallel

    x = tval(1)
    idx = tidx(1)
    do i = 2_c_int, nt
      if (tval(i) > x) then
        x = tval(i)
        idx = tidx(i)
      end if
    end do

    out_value(1) = x
    out_index(1) = idx
  end block
end subroutine argmax_with_index_fp64
