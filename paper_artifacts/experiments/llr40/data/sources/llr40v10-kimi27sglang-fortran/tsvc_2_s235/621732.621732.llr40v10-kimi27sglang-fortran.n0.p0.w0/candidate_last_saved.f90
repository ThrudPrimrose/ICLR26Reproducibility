subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(c, name='tsvc_2_s235_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(*), aa(*)
  real(c_double), intent(in) :: b(*), bb(*), c(*)
  integer :: i, j, n, tid, nthreads, chunk, rem, istart, iend
  integer :: off, prev_off
  n = int(LEN_2D)
  !$omp parallel proc_bind(spread) private(tid, nthreads, chunk, rem, istart, iend, i, j, off, prev_off)
  tid = omp_get_thread_num()
  nthreads = omp_get_num_threads()
  chunk = n / nthreads
  rem = mod(n, nthreads)
  if (tid < rem) then
    istart = tid * (chunk + 1) + 1
    iend = istart + chunk
  else
    istart = tid * chunk + rem + 1
    iend = istart + chunk - 1
  end if
  !$omp simd
  do i = istart, iend
    a(i) = a(i) + b(i) * c(i)
  end do
  !$omp end simd
  prev_off = 0
  off = n
  do j = 2, n
    !$omp simd
    do i = istart, iend
      aa(off + i) = aa(prev_off + i) + bb(off + i) * a(i)
    end do
    !$omp end simd
    prev_off = off
    off = off + n
  end do
  !$omp end parallel
end subroutine tsvc_2_s235_fp64
