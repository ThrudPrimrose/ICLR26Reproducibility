subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(c, name="tsvc_2_s316_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: result(*)
  integer(c_int64_t), value :: LEN_1D
  real(c_double) :: x
  integer(c_int64_t) :: i, j
  integer :: nthreads, tid
  real(c_double), allocatable :: partial(:)

  nthreads = omp_get_max_threads()
  allocate(partial(nthreads))
  partial = huge(1.0_c_double)

  !$omp parallel private(tid, i, x)
  tid = omp_get_thread_num()
  x = huge(1.0_c_double)
  !$omp do schedule(static)
  do i = 1, LEN_1D
     x = min(x, a(i))
  end do
  !$omp end do
  partial(tid+1) = x
  !$omp end parallel

  x = partial(1)
  do j = 2, nthreads
     x = min(x, partial(j))
  end do

  result(1) = x
  deallocate(partial)
end subroutine tsvc_2_s316_fp64
