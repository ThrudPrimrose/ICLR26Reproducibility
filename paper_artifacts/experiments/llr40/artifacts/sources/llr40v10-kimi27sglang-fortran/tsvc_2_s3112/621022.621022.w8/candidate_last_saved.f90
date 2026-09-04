module tsvc_2_s3112_m
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
    integer(c_int64_t), intent(in), value :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: b(LEN_1D)
    integer(c_int64_t) :: i, n, chunk, start2
    real(c_double) :: base
    real(c_long_double), allocatable :: tmp(:)
    real(c_long_double) :: sl

    n = LEN_1D
    if (n <= 4096_c_int64_t) then
      base = 0.0_c_double
      do i = 1_c_int64_t, n
        base = base + a(i)
        b(i) = base
      end do
      return
    end if

    chunk = n / 2_c_int64_t
    start2 = chunk + 1_c_int64_t
    allocate(tmp(start2:n))

    !$omp parallel sections num_threads(2)
    !$omp section
    base = 0.0_c_double
    do i = 1_c_int64_t, chunk
      base = base + a(i)
      b(i) = base
    end do

    !$omp section
    sl = 0.0_c_long_double
    do i = start2, n
      sl = sl + a(i)
      tmp(i) = sl
    end do
    !$omp end parallel sections

    !$omp parallel do schedule(static) private(i)
    do i = start2, n
      b(i) = real(tmp(i) + base, c_double)
    end do
    !$omp end parallel do

    deallocate(tmp)
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112_m
