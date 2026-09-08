subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(c, name="tsvc_2_s3112_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: b(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i, nthreads, chunk, tid, start, end, rem
  real(c_double) :: s, offset
  real(c_double), allocatable :: offsets(:)

  if (LEN_1D <= 1024) then
    s = 0.0_c_double
    do i = 1, LEN_1D
      s = s + a(i)
      b(i) = s
    end do
    return
  end if

  nthreads = 1
  !$ nthreads = omp_get_max_threads()
  chunk = (LEN_1D + nthreads - 1) / nthreads
  allocate(offsets(nthreads))
  offsets = 0.0_c_double

  !$omp parallel private(tid, start, end, i, s)
  tid = 0
  !$ tid = omp_get_thread_num()
  start = tid * chunk + 1
  end = min(start + chunk - 1, LEN_1D)
  if (start <= end) then
    s = 0.0_c_double
    do i = start, end
      s = s + a(i)
      b(i) = s
    end do
    offsets(tid + 1) = s
  end if
  !$omp barrier
  !$omp single
  do i = 2, nthreads
    offsets(i) = offsets(i - 1) + offsets(i)
  end do
  !$omp end single
  !$omp barrier
  if (tid > 0) then
    offset = offsets(tid)
    do i = start, end
      b(i) = b(i) + offset
    end do
  end if
  !$omp end parallel
  deallocate(offsets)
end subroutine tsvc_2_s3112_fp64
