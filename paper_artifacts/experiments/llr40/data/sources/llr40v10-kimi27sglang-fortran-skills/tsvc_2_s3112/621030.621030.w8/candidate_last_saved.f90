subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(LEN_1D)
  integer(c_int64_t) :: nt_max, t, lo, hi, i
  real(c_double) :: run
  real(c_double), allocatable :: part(:)

  nt_max = omp_get_max_threads()
  allocate(part(0:nt_max))
  part(0:nt_max) = 0.0d0

  !$omp parallel private(t, nt_max, lo, hi, i, run)
  nt_max = omp_get_num_threads()
  t = omp_get_thread_num()
  lo = (LEN_1D * t) / nt_max + 1
  hi = (LEN_1D * (t + 1)) / nt_max
  run = 0.0d0
  do i = lo, hi
    run = run + a(i)
    b(i) = run
  end do
  part(t + 1) = run
  !$omp barrier
  !$omp single
  do i = 1, nt_max
    part(i) = part(i) + part(i - 1)
  end do
  !$omp end single
  run = part(t)
  do i = lo, hi
    run = run + a(i)
    b(i) = run
  end do
  !$omp end parallel

  deallocate(part)
end subroutine tsvc_2_s3112_fp64
