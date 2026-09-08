subroutine argmax_with_index_fp64(a, out_index, out_value, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(n)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  real(c_double) :: gmax, lmax
  integer(c_int64_t) :: i, t, nt, lo, hi, gidx, lidx
  real(c_double), allocatable :: tmax(:)
  integer(c_int64_t), allocatable :: tidx(:)

  nt = omp_get_max_threads()
  allocate(tmax(nt), tidx(nt))

  !$omp parallel private(t, lo, hi, i, lmax, lidx)
  t = omp_get_thread_num() + 1
  lo = (n * (t - 1)) / nt + 1
  hi = (n * t) / nt
  if (lo <= hi) then
    lmax = a(lo)
    !$omp simd reduction(max:lmax)
    do i = lo + 1, hi
      if (a(i) > lmax) lmax = a(i)
    end do
    lidx = lo
    do i = lo, hi
      if (a(i) == lmax) then
        lidx = i
        exit
      end if
    end do
    tmax(t) = lmax
    tidx(t) = lidx
  else
    tmax(t) = -huge(1.0_c_double)
    tidx(t) = n + 1
  end if
  !$omp end parallel

  gmax = tmax(1)
  gidx = tidx(1)
  do t = 2, nt
    if (tmax(t) > gmax) then
      gmax = tmax(t)
      gidx = tidx(t)
    end if
  end do

  out_value(1) = gmax
  out_index(1) = gidx
  deallocate(tmax, tidx)
end subroutine argmax_with_index_fp64
