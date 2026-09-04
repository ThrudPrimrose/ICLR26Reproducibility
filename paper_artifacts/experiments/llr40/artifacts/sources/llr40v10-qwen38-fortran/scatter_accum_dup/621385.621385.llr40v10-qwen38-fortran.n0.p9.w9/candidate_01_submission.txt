! scatter_accum_dup:  bins(ip(i)) = bins(ip(i)) + src(i), i = 1..n
! ip values arrive 1-based (harness rebases at the ABI seam) -> subscript directly.
! ip may contain duplicates -> the accumulate is a genuine reduction.
subroutine scatter_accum_dup_fp64(bins, ip, src, len_1d, workspace, ws_size) &
    bind(C, name="scatter_accum_dup_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, ws_size
  real(c_double),       intent(inout)   :: bins(len_1d)
  integer(c_int32_t),   intent(in)      :: ip(len_1d)
  real(c_double),       intent(in)      :: src(len_1d)
  type(c_ptr), value,   intent(in)      :: workspace

  integer :: n, i
  n = int(len_1d)
  if (n <= 0) return

  if (n < 16384) then
    do i = 1, n
      bins(ip(i)) = bins(ip(i)) + src(i)
    end do
    return
  end if

!$omp parallel do schedule(static)
  do i = 1, n
!$omp atomic update
    bins(ip(i)) = bins(ip(i)) + src(i)
!$omp end atomic
  end do
end subroutine scatter_accum_dup_fp64
