subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C, name='tsvc_2_s3112_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: a, b
  integer(c_int64_t), value :: len_1d

  interface
    function sched_getaffinity(pid, cpusetsize, mask) bind(C, name='sched_getaffinity')
      use iso_c_binding
      implicit none
      integer(c_int), intent(in) :: pid
      integer(c_int), intent(in) :: cpusetsize
      integer(c_long), intent(out) :: mask(16)
      integer(c_int) :: sched_getaffinity
    end function sched_getaffinity
    function sched_setaffinity(pid, cpusetsize, mask) bind(C, name='sched_setaffinity')
      use iso_c_binding
      implicit none
      integer(c_int), intent(in) :: pid
      integer(c_int), intent(in) :: cpusetsize
      integer(c_long), intent(in) :: mask(16)
      integer(c_int) :: sched_setaffinity
    end function sched_setaffinity
  end interface

  real(c_double), dimension(:), pointer :: ap, bp
  integer(c_int64_t) :: n, i
  integer(c_long) :: msk(16)
  integer :: rc, cpu, t, r, ncpus
  integer, parameter :: tcs(6) = [24, 24, 24, 48, 48, 24]
  character(len=200) :: lab
  real(c_double) :: s, t0, t1

  n = len_1d
  call c_f_pointer(a, ap, [n])
  call c_f_pointer(b, bp, [n])

  rc = int(sched_getaffinity(int(0, c_int), int(128, c_int), msk), 4)
  write(*,'(A,I0)') 'getaff rc=', int(rc, 4)
  ncpus = 0
  do cpu = 0, 191
    if (ibits(msk(cpu / 64 + 1), mod(cpu, 64) + 1, 1) == 1) ncpus = ncpus + 1
  end do
  write(*,'(A,I0)') 'affinity cpus:', ncpus
  do t = 1, 6
    msk = 0
    select case (t)
    case (1)
      ! current affinity (identity: keep)
      rc = int(sched_getaffinity(int(0, c_int), int(128, c_int), msk), 4)
      lab = 'A_current'
    case (2)
      do cpu = 0, 23; msk(1) = msk(1) + int(1, 8) * ishft(int(1, 8), cpu)
      end do
      lab = 'B_0-23'
    case (3)
      do cpu = 0, 191, 8
        msk(cpu / 64 + 1) = msk(cpu / 64 + 1) + ishft(int(1, 8), mod(cpu, 64))
      end do
      lab = 'C_spread_8step'
    case (4)
      do cpu = 48, 95; msk(1) = msk(1) + ishft(int(1, 8), mod(cpu, 64))
      end do
      lab = 'D_48-95(48thr)'
    case (5)
      do cpu = 0, 95, 4
        msk(cpu / 64 + 1) = msk(cpu / 64 + 1) + ishft(int(1, 8), mod(cpu, 64))
      end do
      lab = 'E_0-95step4(48thr)'
    case (6)
      do cpu = 0, 71, 3
        msk(cpu / 64 + 1) = msk(cpu / 64 + 1) + ishft(int(1, 8), mod(cpu, 64))
      end do
      lab = 'F_0-71step3'
    end select
    rc = sched_setaffinity(int(0, c_int), int(128, c_int), msk)
    call omp_set_num_threads(tcs(t))
    t0 = omp_get_wtime()
    do r = 1, 2
      s = 0.0d0
      !$omp parallel do reduction(+:s)
      do i = 1, n
        s = s + ap(i)
        bp(i) = ap(i) * 1.0d0
      end do
      t1 = omp_get_wtime()
    end do
    write(*,'(A,A12,A,I3,A,F8.3)') lab, ' rc=', int(rc, 4), ' thr=', tcs(t), ' bw_GBps=', dble(3*n)*8.0d0/(t1-t0)/1.0d9
  end do
  flush(6)
end subroutine
