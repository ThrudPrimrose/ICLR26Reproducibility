subroutine scan_affine_decay(c, x, y, n, workspace, workspace_size) bind(C, name='scan_affine_decay_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in), value :: n
  real(c_double), intent(in) :: c(n)
  real(c_double), intent(in) :: x(n)
  real(c_double), intent(inout) :: y(n)
  type(c_ptr), intent(in), value :: workspace
  integer(c_int64_t), intent(in), value :: workspace_size
  integer(c_int64_t) :: i, nt, m, t, lo, hi
  real(c_double) :: run, run_c, run_x, seed
  real(c_double), allocatable :: lc(:), lx(:), gc(:), gx(:)

  if (n <= 0) return
  y(1) = x(1)
  if (n == 1) return

  nt = omp_get_max_threads()
  if (nt <= 1 .or. n < 4096) then
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  m = n - 1
  allocate(lc(nt), lx(nt), gc(nt), gx(nt))
  gc(1) = 1.0d0
  gx(1) = 0.0d0

  !$omp parallel default(none) shared(c, x, y, n, m, nt, lc, lx, gc, gx) &
  !$omp private(i, run_c, run_x, run, seed, t, lo, hi)
  t = omp_get_thread_num()
  lo = 2 + (m * t) / nt
  hi = 1 + (m * (t + 1)) / nt

  run_c = 1.0d0
  run_x = 0.0d0
  do i = lo, hi
    run_c = run_c * c(i)
    run_x = c(i) * run_x + x(i)
  end do
  lc(t + 1) = run_c
  lx(t + 1) = run_x

  !$omp barrier

  !$omp single
  do i = 2, nt
    gc(i) = lc(i - 1) * gc(i - 1)
    gx(i) = lc(i - 1) * gx(i - 1) + lx(i - 1)
  end do
  !$omp end single

  !$omp barrier

  if (lo <= hi) then
    seed = gc(t + 1) * x(1) + gx(t + 1)
    run = c(lo) * seed + x(lo)
    y(lo) = run
    do i = lo + 1, hi
      run = c(i) * run + x(i)
      y(i) = run
    end do
  end if
  !$omp end parallel

  deallocate(lc, lx, gc, gx)
end subroutine scan_affine_decay
