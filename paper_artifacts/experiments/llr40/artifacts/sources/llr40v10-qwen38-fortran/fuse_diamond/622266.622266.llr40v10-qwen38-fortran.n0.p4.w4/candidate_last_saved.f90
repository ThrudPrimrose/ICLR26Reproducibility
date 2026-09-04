subroutine fuse_diamond_fp64(a, out, n) bind(c, name='fuse_diamond_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use omp_lib, only: omp_get_wtime, omp_get_max_threads
  implicit none
  integer(c_int64_t), value :: n
  real(c_double), dimension(n), intent(in) :: a
  real(c_double), dimension(n), intent(out) :: out
  real(c_double) :: t0, t1
  integer :: nthr

  nthr = omp_get_max_threads()

  t0 = omp_get_wtime()
  call run_kernel(a, out, n)
  t1 = omp_get_wtime()
  write(*, '(A,I0,A,F12.2,A,I0)') ' KTIME_MS=', n, ' MS=', (t1 - t0) * 1.0d6, ' NTHR=', nthr
  flush(0)

contains

  subroutine run_kernel(a, out, n)
    implicit none
    integer(c_int64_t), intent(in) :: n
    real(c_double), dimension(n), intent(in) :: a
    real(c_double), dimension(n), intent(out) :: out
    integer(c_int64_t) :: i
    real(c_double) :: t

    !omp parallel do default(none) shared(a, out, n) private(i, t)
    do i = 1, n
      t = a(i) * a(i)
      out(i) = (t + 1.0d0) * (t - 1.0d0)
    end do
  end subroutine run_kernel
end subroutine fuse_diamond_fp64
