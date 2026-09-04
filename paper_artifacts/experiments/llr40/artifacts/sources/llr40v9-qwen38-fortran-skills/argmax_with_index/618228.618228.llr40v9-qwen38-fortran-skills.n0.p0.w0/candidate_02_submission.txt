subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  real(c_double), intent(inout) :: workspace(workspace_size)

  integer :: nt, t
  integer(c_int64_t) :: lo, hi, i, nmain, nmain4
  real(c_double) :: v0, v1, v2, v3, v
  integer(c_int64_t) :: k0, k1, k2, k3, k
  real(c_double), allocatable :: pval(:)
  integer(c_int64_t), allocatable :: pidx(:)

  if (len_1d == 1) then
    out_value(1) = a(1)
    out_index(1) = 1
    return
  end if

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  allocate(pval(nt), pidx(nt))
  pval = -huge(0.0d0)
  pidx = huge(1_c_int64_t)

  !$omp parallel shared(a, pval, pidx, nt, len_1d) private(t, lo, hi, i, nmain, nmain4, v0,v1,v2,v3, v, k0,k1,k2,k3, k)
  t = omp_get_thread_num()
  hi = (len_1d * t) / nt + 1
  lo = (len_1d * (t + 1)) / nt
  if (hi <= lo) then
  v0 = -huge(0.0d0); v1 = -huge(0.0d0); v2 = -huge(0.0d0); v3 = -huge(0.0d0)
  k0 = huge(1_c_int64_t); k1 = huge(1_c_int64_t); k2 = huge(1_c_int64_t); k3 = huge(1_c_int64_t)
  nmain = lo - hi + 1
  nmain4 = (nmain / 4) * 4
  do i = hi, hi + nmain4 - 1, 4
    if (a(i)   > v0) then; v0 = a(i);   k0 = i;   end if
    if (a(i+1) > v1) then; v1 = a(i+1); k1 = i+1; end if
    if (a(i+2) > v2) then; v2 = a(i+2); k2 = i+2; end if
    if (a(i+3) > v3) then; v3 = a(i+3); k3 = i+3; end if
  end do
  do i = hi + nmain4, lo
    if (a(i) > v0) then; v0 = a(i); k0 = i; end if
  end do
  ! combine the 4 stride-class accumulators (min index on ties)
  v = v0; k = k0
  if (v1 > v) then; v = v1; k = k1; else if (v1 == v .and. k1 < k) then; k = k1; end if
  if (v2 > v) then; v = v2; k = k2; else if (v2 == v .and. k2 < k) then; k = k2; end if
  if (v3 > v) then; v = v3; k = k3; else if (v3 == v .and. k3 < k) then; k = k3; end if
  pval(t + 1) = v
  pidx(t + 1) = k
  end if
  !$omp end parallel

  out_value(1) = pval(1)
  out_index(1) = pidx(1)
  do t = 2, nt
    if (pval(t) > out_value(1)) then
      out_value(1) = pval(t)
      out_index(1) = pidx(t)
    else if (pval(t) == out_value(1) .and. pidx(t) < out_index(1)) then
      out_index(1) = pidx(t)
    end if
  end do
  deallocate(pval, pidx)
end subroutine argmax_with_index_fp64
