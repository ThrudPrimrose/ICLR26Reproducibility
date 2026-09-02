subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S, workspace, workspace_size) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: S
  integer(c_int8_t), intent(in) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: n, i, ntiles, it, tstart, tstop, tlo, thi
  integer :: nt, tid
  logical :: fast
  integer(c_int64_t) :: t
  real(c_double), allocatable, target, save :: sbuf_ser(:)
  real(c_double), allocatable, target :: tbp(:)
  real(c_double), pointer :: tb(:)
  integer(c_int64_t), parameter :: I64MIN = int(-2_c_int64_t)**63

  n = LEN_1D
  if (n <= 0_c_int64_t) return

  ! For S in {0, +/-1, +/-2**k} the product S*b is exact (one rounding), so a
  ! fused multiply-add yields the same bits as multiply-then-add and the whole
  ! update is one vector op per element. For other S the product must be
  ! materialized before the add (no FMA contraction), which the strict tile
  ! does through a per-thread scratch row.
  if (S == 0_c_int64_t) then
    fast = .true.
  else if (S > 0_c_int64_t) then
    fast = iand(S, S - 1_c_int64_t) == 0_c_int64_t
  else if (S == I64MIN) then
    fast = .true.
  else
    t = -S
    fast = iand(t, t - 1_c_int64_t) == 0_c_int64_t
  end if

  if (fast) then
    if (n < 2097152_c_int64_t) then
      do i = 1, n
        a(i) = a(i) + S*b(i)
      end do
    else
      nt = max(omp_get_max_threads(), 1)
      !$omp parallel do schedule(static) num_threads(nt)
      do i = 1, n
        a(i) = a(i) + S*b(i)
      end do
      !$omp end parallel do
    end if
  else
    if (n < 2097152_c_int64_t) then
      if (.not. allocated(sbuf_ser)) allocate(sbuf_ser(131072))
      tb => sbuf_ser
      do i = 1, n, 131072_c_int64_t
        call strict_tile(a, b, tb, i, min(131072_c_int64_t, n - i + 1), S)
      end do
    else
      nt = max(omp_get_max_threads(), 1)
      ntiles = (n + 131071_c_int64_t) / 131072_c_int64_t
      !$omp parallel num_threads(nt) private(tid, tlo, thi, it, tstart, tstop, tb, tbp)
        allocate(tbp(131072))
        tb => tbp
        tid = omp_get_thread_num()
        tlo = (ntiles / nt) * tid + 1
        thi = (ntiles / nt) * (tid + 1_c_int64_t)
        if (tid == nt - 1) thi = ntiles
        do it = tlo, thi
          tstart = (it - 1_c_int64_t) * 131072_c_int64_t + 1
          tstop = min(it * 131072_c_int64_t, n)
          call strict_tile(a, b, tb, tstart, tstop - tstart + 1, S)
        end do
        deallocate(tbp)
      !$omp end parallel
    end if
  end if
contains
  subroutine strict_tile(a, b, tb, start, len, S)
    implicit none
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(inout) :: tb(*)
    integer(c_int64_t), value, intent(in) :: start
    integer(c_int64_t), value, intent(in) :: len
    integer(c_int64_t), value, intent(in) :: S
    integer(c_int64_t) :: j
    do j = 1, len
      tb(j) = S*b(start + j - 1)
    end do
    do j = 1, len
      a(start + j - 1) = a(start + j - 1) + tb(j)
    end do
  end subroutine strict_tile
end subroutine tsvc_2_vpvts_fp64
