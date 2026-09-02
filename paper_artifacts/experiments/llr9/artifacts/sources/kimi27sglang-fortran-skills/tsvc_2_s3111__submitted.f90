subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(2)
  real(c_double) :: sum_val
  real(c_double), allocatable :: part(:)
  integer(c_int64_t) :: i, lo, hi
  integer :: nt, t
  real(c_double) :: run

  nt = omp_get_max_threads()
  allocate(part(nt))
  part = 0.0_c_double

!$omp parallel private(t, lo, hi, i, run)
  t = omp_get_thread_num()
  lo = (LEN_1D * int(t, c_int64_t)) / int(nt, c_int64_t) + 1_c_int64_t
  hi = (LEN_1D * int(t + 1, c_int64_t)) / int(nt, c_int64_t)
  run = 0.0_c_double
  do i = lo, hi
    if (a(i) > 0.0_c_double) then
      run = run + a(i)
    end if
  end do
  part(t + 1) = run
!$omp end parallel

  sum_val = sum(part)
  deallocate(part)

  b(1) = sum_val
end subroutine tsvc_2_s3111_fp64
