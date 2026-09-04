subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(c, name="tsvc_2_s1232_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_int, c_uint, c_size_t, c_char, c_ptr
  implicit none
  integer(c_int64_t), value :: len_2d, vlen
  real(c_double), intent(inout), dimension(*) :: aa
  real(c_double), intent(in), dimension(*) :: bb
  real(c_double), intent(in), dimension(*) :: cc

  interface
    type(c_ptr) function c_fopen(path, mode) bind(c, name="fopen")
      import :: c_ptr, c_char
      character(kind=c_char), intent(in) :: path(*), mode(*)
    end function
    integer(c_int) function c_fclose(fp) bind(c, name="fclose")
      import :: c_int, c_ptr
      type(c_ptr), value :: fp
    end function
    type(c_ptr) function c_fgets(s, n, fp) bind(c, name="fgets")
      import :: c_int, c_char, c_ptr
      character(kind=c_char), intent(out) :: s(*)
      integer(c_int), value :: n
      type(c_ptr), value :: fp
    end function
    integer(c_int) function c_sched_setaffinity(pid, size, mask) bind(c, name="sched_setaffinity")
      import :: c_uint, c_size_t, c_ptr
      integer(c_uint), value :: pid
      integer(c_size_t), value :: size
      type(c_ptr), value :: mask
    end function
    integer(c_int) function c_omp_set_num_threads(n) bind(c, name="omp_set_num_threads")
      import :: c_int
      integer(c_int), value :: n
    end function
  end interface

  integer(c_int64_t) :: n64, v64
  integer :: n, v, q, k, jp, r
  integer(c_int64_t), allocatable :: wsum(:)
  integer(c_int64_t) :: wtot, target
  type(c_ptr) :: fp
  character(kind=c_char) :: lbuf(1024)
  integer(c_int64_t) :: addrs(3), vma
  integer :: nl, nodecnt(0:15), dom(3), node, cnt, nc
  integer :: cpus(1024), ncpu, ncpu_eff, cpus_eff(1024), tmax
  integer :: t, tid, qlo, qhi, ilos, ihi
  integer(c_uint), target :: amask(128)
  logical :: lpin, allsame
  character(kind=c_char) :: path(64), nodelist(256)
  integer :: c, pos, start, end, ispace, eq, d

  n64 = len_2d
  v64 = vlen
  n = int(n64)
  v = int(vlen)
  if (n < 1) return

  ! ---- total work per column q: (q-1)/v + 1 (v>0) else n
  allocate(wsum(n + 1))
  wsum(0) = 0
  do q = 1, n
     if (v > 0) then
        wsum(q) = wsum(q - 1) + (q - 1) / v + 1
     else
        wsum(q) = wsum(q - 1) + n
     end if
  end do
  wtot = wsum(n)
  if (wtot <= 0) then
     deallocate(wsum)
     return
  end if

  ! ---- default plan: 24 threads, no pinning
  tmax = 24
  lpin = .false.
  ncpu = 0
  call read_cpulist("/sys/devices/system/node/node0/cpulist", cpus, ncpu)
  call read_cpulist("/sys/fs/cgroup/cpuset.cpus.effective", cpus_eff, ncpu_eff)
  if (ncpu > 0 .and. ncpu_eff > 0) then
     do c = 1, ncpu
        if (int(cpu_in(cpus(c), cpus_eff, ncpu_eff)) == 1) cpus(c) = -1
     end do
  end if

  ! ---- probe NUMA node of the data via /proc/self/numa_maps
  nodecnt = 0
  allsame = .true.
  do c = 1, 3
     dom(c) = -1
  end do
  fp = c_fopen("/proc/self/numa_maps", "r")
  if (.not. c_ptr_null(fp)) then
     nl = 0
     do
        fp2: call fgetline(fp, lbuf)
        if (c_ptr_null(fp2_result)) exit fp2
        nl = nl + 1
        if (nl > 2000) exit
     end do
  end if

  ! collect lines into a small array (re-scan)
  character(kind=c_char), allocatable :: lines(:,:)
  if (.not. c_ptr_null(fp)) then
     call c_fgets... 
  end if
  ...
end subroutine
