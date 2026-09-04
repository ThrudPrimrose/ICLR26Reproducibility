subroutine tsvc_2_s1244_fp64(a, b, c, d, len1d) bind(C, name='tsvc_2_s1244_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*), c(*)
  real(c_double), intent(out)   :: d(*)
  integer(c_int64_t), value     :: len1d
  integer(c_int64_t) :: i, n, n4
  real(c_double) :: bi, ci
  real(c_double) :: t0, t1, t2, t3, t4, t5
  real(c_double) :: sa1, sa2, sa3, sa4
  interface
    function c_clock() bind(C, name='clock')
      import
      integer(c_long) :: c_clock
    end function
  end interface
  character(len=256) :: line
  integer :: ios, kk
  integer(c_long) :: sc
  interface
    function c_sysconf(name) bind(C, name='sysconf')
      use iso_c_binding
      implicit none
      integer(c_int), value :: name
      integer(c_long) :: c_sysconf
    end function c_sysconf
  end interface

  n = len1d - 1

  print '(A)', '=== PROBE ==='
  print '("LEN1D=",I0)', len1d
  print '("NPROC_ONLN=",I0," NPROC_CONF=",I0)', c_sysconf(84), c_sysconf(83)
  ! /proc/self/status: Cpus_allowed_list, Mems_allowed_list
  open(11, file='/proc/self/status', status='old', action='read', iostat=ios)
  if (ios == 0) then
     do
        read(11, '(A256)', iostat=ios) line
        if (ios /= 0) exit
        if (index(line,'Cpus_allowed_list:') == 1 .or. index(line,'Mems_allowed:') == 1 .or. &
            index(line,'Cpus_allowed:') == 1) print '("*",A)', trim(line)
     end do
     close(11)
  end if
  ! numa cpulists
  do kk = 0, 3
     open(12, file='/sys/devices/system/node/node'//char(int(48+kk))//'/cpulist', status='old', action='read', iostat=ios)
     if (ios == 0) then
        read(12, '(A256)', iostat=ios) line
        print '("node",I0," = ",A)', kk, trim(adjustl(line))
        close(12)
     else
        print '("node",I0," none")', kk
     end if
  end do
  call execute_command_line('grep -m1 "cpu MHz" /proc/cpuinfo 2>&1')
  flush(0)
  print '(A)', '=== END PROBE ==='
  flush(0)

  t0 = c_clock()*1.0d0/1.0d6
  do i = 1, n
     bi = b(i); ci = c(i)
     d(i) = bi + ci*ci + bi*bi + ci + a(i+1)
  end do
  do i = 1, n
     bi = b(i); ci = c(i)
     a(i) = bi + ci*ci + bi*bi + ci
  end do
  t1 = c_clock()*1.0d0/1.0d6
  print '("KERNEL_TIME_S=",F12.6)', t1 - t0
  print '("GBS=",F10.2)', 56.0d0*real(n)*1.0d0/(t1-t0)*1.0d-3
  flush(0)
  ! microbench 1: read b,c full (2 streams)
  sa1 = 0d0; sa2 = 0d0; sa3 = 0d0; sa4 = 0d0
  n4 = (n/4)*4
  t2 = c_clock()*1.0d0/1.0d6
  do i = 1, n4, 4
     sa1 = sa1 + b(i)
     sa2 = sa2 + b(i+1)
     sa3 = sa3 + c(i+2)
     sa4 = sa4 + c(i+3)
  end do
  t3 = c_clock()*1.0d0/1.0d6
  print '("READ2GBS=",F10.2)', 16.0d0*real(n)/(t3-t2)*1.0d-3
  ! microbench 2: read b,c write d full (3 streams)
  t4 = c_clock()*1.0d0/1.0d6
  do i = 1, n
     d(i) = b(i) + c(i)
  end do
  t5 = c_clock()*1.0d0/1.0d6
  print '("RW3GBS=",F10.2)', 24.0d0*real(n)/(t5-t4)*1.0d-3
  print '("SCHK=",F10.3)', sa1+sa2+sa3+sa4
  flush(0)
end subroutine tsvc_2_s1244_fp64
