subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C, name="tsvc_2_s3112_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(len_1d)
  integer(c_int64_t) :: n, i, blo, bhi, nblk, iblk
  integer :: nt, t, nw
  double precision :: run
  double precision :: part(0:24)
  n = len_1d
  if (n <= 0) return
  nt = omp_get_max_threads()
  ! serial fast path: bit-identical to the reference scan
  if (n < 65536) then
    run = 0.0d0
    do i = 1, n
      run = run + a(i)
      b(i) = run
    end do
    return
  end if
  ! tile the array into slabs of nt blocks so each slab stays cache-resident
  nw = 1
  if (n >= int(1048576, 8)) nw = 16
  nblk = int(nw, 8) * int(nt, 8)
  part(0) = 0.0d0
  do nw = 0, int(16, 8) - 1
    if (nblk < 2) exit
    !$omp parallel default(shared) private(t, blo, bhi, i, run, iblk)
    t = omp_get_thread_num()
    if (t >= nt) cycle
    iblk = nw * nt + t
    blo = n * iblk / nblk + 1
    bhi = n * (iblk + 1) / nblk
    run = 0.0d0
    !$omp simd reduction(+:run)
    do i = blo, bhi
      run = run + a(i)
    end do
    part(t + 1) = run
    !$omp end parallel
    do t = 1, nt
      part(t) = part(t) + part(t - 1)
    end do
    !$omp parallel default(shared) private(t, blo, bhi, i, run, iblk)
    t = omp_get_thread_num()
    if (t >= nt) cycle
    iblk = nw * nt + t
    blo = n * iblk / nblk + 1
    bhi = n * (iblk + 1) / nblk
    run = part(t)
    do i = blo, bhi
      run = run + a(i)
      b(i) = run
    end do
    !$omp end parallel
  end do
end subroutine tsvc_2_s3112_fp64
