subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib, only: omp_get_thread_num
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(LEN_1D)
  real(c_double) :: part(0:2)
  integer :: t
  integer(c_int64_t) :: i, lo, hi, mid
  real(c_double) :: run

  if (LEN_1D < 4096_c_int64_t) then
    run = 0.0_c_double
    do i = 1, LEN_1D
      run = run + a(i)
      b(i) = run
    end do
    return
  end if

  part = 0.0_c_double
  mid = LEN_1D / 2

  !$omp parallel num_threads(2) default(none) private(t, i, lo, hi, run) &
  !$omp shared(a, b, part, LEN_1D, mid)
  t = omp_get_thread_num()
  if (t == 0) then
    lo = 1
    hi = mid
    run = 0.0_c_double
    do i = lo, hi
      run = run + a(i)
      b(i) = run
    end do
    part(1) = run
  else
    lo = mid + 1
    hi = LEN_1D
    run = 0.0_c_double
    do i = lo, hi
      run = run + a(i)
    end do
    part(2) = run
  end if

  !$omp barrier

  if (t == 1) then
    run = part(1)
    do i = lo, hi
      run = run + a(i)
      b(i) = run
    end do
  end if
  !$omp end parallel
end subroutine tsvc_2_s3112_fp64
