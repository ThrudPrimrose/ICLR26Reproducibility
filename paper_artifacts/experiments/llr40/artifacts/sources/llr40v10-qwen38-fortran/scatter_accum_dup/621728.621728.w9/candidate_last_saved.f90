! scatter_accum_dup -- bins(ip(i)) += src(i); ip values are 1-based (Fortran base),
! ip may contain duplicates -> indexed accumulate with possible conflicts.
!
! Strategy:
!   * small n: tight serial loop (no OpenMP overhead)
!   * else: workspace-backed int8 occurrence counter (zeroed per rep by the
!     harness), histogram it with atomic updates, detect permutation via a max
!     scan, then scatter plain (permutation) or with atomic updates (duplicates).
subroutine scatter_accum_dup_fp64(bins, ip, src, len_1d, ws, ws_bytes) bind(C, name='scatter_accum_dup_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: bins(*)
  integer(c_int32_t), intent(in) :: ip(*)
  real(c_double), intent(in)    :: src(*)
  integer(c_int64_t), value     :: len_1d
  type(c_ptr), value            :: ws
  integer(c_int64_t), value     :: ws_bytes

  integer(c_int8_t), pointer     :: cbp(:)
  integer(c_int8_t), allocatable, target :: cba(:)
  integer(c_int64_t) :: n, i
  integer(c_int8_t) :: mx
  logical :: have_ws

  n = len_1d

  if (n <= 65536) then
    do i = 1, n
      bins(ip(i)) = bins(ip(i)) + src(i)
    end do
    return
  end if

  have_ws = c_associated(ws) .and. ws_bytes >= n
  if (have_ws) then
    call c_f_pointer(ws, cbp, [n])
  else
    allocate(cba(n))
    cba = 0_c_int8_t
    cbp => cba
  end if

  !$omp parallel default(none) shared(bins, ip, src, cbp, n, mx) private(i)
    !$omp do schedule(static)
    do i = 1, n
      !$omp atomic update
      cbp(ip(i)) = cbp(ip(i)) + 1_c_int8_t
    end do
    !$omp end do

    mx = 0_c_int8_t
    !$omp do reduction(max:mx) schedule(static)
    do i = 1, n
      mx = max(mx, cbp(i))
    end do
    !$omp end do

    if (mx == 1_c_int8_t) then
      !$omp do schedule(static)
      do i = 1, n
        bins(ip(i)) = bins(ip(i)) + src(i)
      end do
      !$omp end do
    else
      !$omp do schedule(static)
      do i = 1, n
        !$omp atomic update
        bins(ip(i)) = bins(ip(i)) + src(i)
      end do
      !$omp end do
    end if
  !$omp end parallel

  if (.not. have_ws) then
    deallocate(cba)
  end if
end subroutine scatter_accum_dup_fp64
