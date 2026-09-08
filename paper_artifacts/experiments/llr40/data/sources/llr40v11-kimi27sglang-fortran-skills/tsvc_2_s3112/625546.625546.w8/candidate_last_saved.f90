subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name='tsvc_2_s3112_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)

  integer :: nthreads, k, t
  integer(c_int64_t) :: i, lo, hi
  real(c_double) :: run, offset
  real(c_double), allocatable :: partial(:)

  nthreads = omp_get_max_threads()
  k = min(nthreads, 4)
  if (k < 1) k = 1

  allocate(partial(0:k))
  partial = 0.0_c_double

  !$omp parallel private(t, lo, hi, i, run) shared(a, b, partial, LEN_1D, k) num_threads(k)
  t = omp_get_thread_num()
  lo = (LEN_1D * int(t, c_int64_t)) / k + 1
  hi = (LEN_1D * int(t + 1, c_int64_t)) / k

  run = 0.0_c_double
  do i = lo, hi
    run = run + a(i)
    b(i) = run
  end do
  partial(t + 1) = run
  !$omp end parallel

  do i = 1, k
    partial(i) = partial(i) + partial(i - 1)
  end do

  !$omp parallel private(t, lo, hi, i, offset) shared(b, partial, LEN_1D, k) num_threads(k)
  t = omp_get_thread_num()
  lo = (LEN_1D * int(t, c_int64_t)) / k + 1
  hi = (LEN_1D * int(t + 1, c_int64_t)) / k
  offset = partial(t)

  do i = lo, hi
    b(i) = b(i) + offset
  end do
  !$omp end parallel

  deallocate(partial)
end subroutine tsvc_2_s3112_fp64
