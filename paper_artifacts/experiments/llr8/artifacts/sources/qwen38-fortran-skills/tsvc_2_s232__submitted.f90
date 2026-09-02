subroutine tsvc_2_s232_fp64(aa, bb, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)

  integer(c_int64_t) :: n, nt, t, total, lo, hi, il, ih, mid, jj, ii
  real(c_double) :: x

  n = len_2d
  if (n < 2) return
  total = n * (n - 1) / 2
  nt = int(omp_get_max_threads(), c_int64_t)

  !$omp parallel shared(aa, bb, n, nt, total) &
!$omp& private(t, lo, hi, il, ih, mid, jj, ii, x)
  t = int(omp_get_thread_num(), c_int64_t)
  ! thread t owns row jj when floor(A(jj)*nt/total) == t,
  ! where A(jj) = (jj-1)*(jj-2)/2 = work strictly before row jj
  if (t * total == 0) then
    lo = 2
  else
    ! smallest jj in [2,n] with A(jj)*nt >= t*total, or n+1 if none
    il = 2; ih = n
    do while (il < ih)
      mid = (il + ih) / 2
      if ((mid - 1) * (mid - 2) / 2 * nt >= t * total) then
        ih = mid
      else
        il = mid + 1
      end if
    end do
    lo = il
    if ((n - 1) * (n - 2) / 2 * nt < t * total) lo = n + 1
  end if
  ! largest jj in [2,n] with A(jj)*nt < (t+1)*total
  il = 2; ih = n
  do while (il < ih)
    mid = (il + ih + 1) / 2
    if ((mid - 1) * (mid - 2) / 2 * nt < (t + 1) * total) then
      il = mid
    else
      ih = mid - 1
    end if
  end do
  hi = il

  do jj = lo, hi
    x = aa(1, jj)
    do ii = 1, jj - 1
      x = x * x + bb(ii + 1, jj)
      aa(ii + 1, jj) = x
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s232_fp64
