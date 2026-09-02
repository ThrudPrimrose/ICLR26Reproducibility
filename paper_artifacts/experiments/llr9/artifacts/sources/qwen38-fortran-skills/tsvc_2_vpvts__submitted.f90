subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, s, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  real(c_double) :: sf
  integer(c_int64_t) :: i64
  integer :: i, nt, cnt, tid
  integer, save :: eff_cores = 0
  integer :: cpulist(256)

  interface
    function sched_getcpu() bind(C, name='sched_getcpu')
      import :: c_int
      integer(c_int) :: sched_getcpu
    end function
  end interface

  if (len_1d <= 0 .or. s == 0) return
  sf = real(s, c_double)

  if (len_1d > 2147483647_8) then
    ! absurdly large size: plain threaded pass
    !$omp parallel do
    do i64 = 1, len_1d
      a(i64) = a(i64) + b(i64) * sf
    end do
  else if (len_1d > 524288_8) then
    ! large streaming path
    if (eff_cores == 0) then
      ! one-shot probe: count distinct physical CPUs the runtime puts us on
      nt = min(omp_get_max_threads(), 256)
      if (nt <= 1) then
        eff_cores = 1
      else
      cpulist = -1
      !$omp parallel
      tid = omp_get_thread_num()
      if (tid < nt) cpulist(tid + 1) = sched_getcpu()
      !$omp end parallel
      cnt = 0
      do i = 1, nt
        if (cnt == 0) then
          cnt = 1
          cpulist(cnt) = cpulist(i)
        else
          if (all(cpulist(1:cnt) /= cpulist(i))) then
            cnt = cnt + 1
            cpulist(cnt) = cpulist(i)
          end if
        end if
      end do
        eff_cores = max(cnt, 1)
        if (eff_cores < nt) call omp_set_num_threads(eff_cores)
      end if
    end if
    if (eff_cores <= 1) then
      ! single effective core: no team spawn, plain vectorized loop
      !$omp simd
      do i = 1, int(len_1d)
        a(i) = a(i) + b(i) * sf
      end do
    else
      !$omp parallel do simd
      do i = 1, int(len_1d)
        a(i) = a(i) + b(i) * sf
      end do
    end if
  else
    ! small: one core streaming, no team
    !$omp simd
    do i = 1, int(len_1d)
      a(i) = a(i) + b(i) * sf
    end do
  end if
end subroutine tsvc_2_vpvts_fp64
