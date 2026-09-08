! tsvc_2_vpvts -- a(i) = a(i) + b(i)*S  (fp64)
!
! Strategy:
!   n >= 8M  : 24 persistent pthread workers executing embedded AVX-512
!              machine code (32-element unroll, non-temporal stores with a
!              per-chunk 256K-element coherent tail), one worker per CPU,
!              CPUs ordered by the NUMA node owning the data, found with a
!              one-time runtime read-bandwidth probe.
!   n >= 16K : OpenMP static-parallel (identical to the simple baseline).
!   n < 16K  : serial vectorized loop (beats thread-spawn overhead).
module kernmod
  use iso_c_binding
  implicit none
  private
  integer, parameter :: i8 = 8
  integer, parameter :: ntw = 24
  integer, parameter :: nblk = 80
  integer, parameter :: blobsize = 640_8

  type :: slot_t
    integer(i8) :: a
    integer(i8) :: b
    integer(i8) :: lo
    integer(i8) :: hi
    real(kind=8) :: sd
    integer(i8) :: tid
    integer(i8) :: seen
    integer(i8) :: sh
  end type slot_t

  type :: g_t
    integer(i8) :: seq
    integer(i8) :: done
    type(slot_t) :: sl(0:ntw-1)
  end type g_t

  ! embedded x86-64 AVX-512 worker code (leaf, infinite loop, rdi = slot)
  integer(i8), target :: blob(nblk) = [&
z'5553415441554156',
    z'41574989FF488B7F',
    z'3849C7C300000000',
    z'4C8B274D3B67300F',
    z'8411020000498B07',
    z'498B5F084D8B6710',
    z'4D8B6F18C4C17B10',
    z'472062F2FD4819C0',
    z'4E8D34E049F7C60F',
    z'00000074334D39EC',
    z'0F8DBD010000F242',
    z'0F100CE0F2420F10',
    z'14E3F20F59D0F20F',
    z'58CAF2420F110CE0',
    z'49FFC44E8D34E049',
    z'F7C60F00000075CD',
    z'4D8D950000FCFF4D',
    z'39E24D0F4CD449F7',
    z'C40700000074284D',
    z'39D40F8D15010000',
    z'F2420F100CE0F242',
    z'0F1014E3F20F59D0',
    z'F20F58CAF2420F11',
    z'0CE049FFC4EBD84D',
    z'89E94983E9484D8D',
    z'74241F4D39D60F8D',
    z'AD0000004D39CC77',
    z'12420F1884E00002',
    z'0000420F1884E300',
    z'02000062B1FD4810',
    z'0CE062B1FD481014',
    z'E362B1FD48105CE0',
    z'0162B1FD481064E3',
    z'0162F1ED4859D062',
    z'F1DD4859E062F1F5',
    z'4858CA62F1E54858',
    z'DC62B1FD482B0CE0',
    z'62B1FD482B5CE001',
    z'62B1FD48106CE002',
    z'62B1FD481074E302',
    z'62B1FD48107CE003',
    z'6231FD481044E303',
    z'62F1CD4859F06271',
    z'BD4859C062F1D548',
    z'58EE62D1C54858F8',
    z'62B1FD482B6CE002',
    z'62B1FD482B7CE003',
    z'4983C420E93EFFFF',
    z'FF4D8D7424074D39',
    z'D67D2A62B1FD4810',
    z'0CE062B1FD481014',
    z'E362F1ED4859D062',
    z'F1F54858CA62B1FD',
    z'482B0CE04983C408',
    z'E90AFFFFFF49F7C4',
    z'0700000074244D39',
    z'EC7D50F2420F100C',
    z'E0F2420F1014E3F2',
    z'0F59D0F20F58CAF2',
    z'420F110CE049FFC4',
    z'EBDC4D8D7424074D',
    z'39EE7DD262B1FD48',
    z'100CE062B1FD4810',
    z'14E362F1ED4859D0',
    z'62F1F54858CA62B1',
    z'FD48110CE04983C4',
    z'08EBCF0FAEF84C8B',
    z'374D89773049C7C6',
    z'01000000F04C0FC1',
    z'770849C7C3000400',
    z'00E9E2FDFFFF4D85',
    z'DB75344989E24983',
    z'E2F049C742F00000',
    z'000049C742F8D007',
    z'0000498D7AF048C7',
    z'C0E600000048C7C6',
    z'000000000F05498B',
    z'7F38E9A9FDFFFF49',
    z'FFCBF390E99FFDFF',
    z'FF00000000000000'
  ]

  integer(i8) :: mask32(16) = 0_8
  type(c_ptr) :: blob_ptr = c_null_ptr
  type(g_t), target :: g
  integer(i8) :: pin_cpu(ntw) = 0_8
  integer :: g_nw = 0
  logical :: g_ready = .false.
  logical :: g_failed = .false.
  logical :: g_dbg = .false.
  real(kind=8) :: g_probe_acc = 0.0_8
  integer :: g_spin = 0
  integer :: g_wt_printed = 0


  interface
    function c_sched_getaffinity(pid, size, mask) bind(c, name='sched_getaffinity')
      import :: i8
      integer(i8), value :: pid, size
      integer, intent(out) :: mask(16)
      integer :: c_sched_getaffinity
    end function c_sched_getaffinity
    function c_sched_setaffinity(pid, size, mask) bind(c, name='sched_setaffinity')
      import :: i8
      integer(i8), value :: pid, size
      integer, intent(in) :: mask(16)
      integer :: c_sched_setaffinity
    end function c_sched_setaffinity
    function c_pthread_create(tid, attr, start, arg) bind(c, name='pthread_create')
      import :: i8
      integer(i8), intent(out) :: tid
      type(c_ptr), value :: attr
      type(c_funptr), value :: start
      integer(i8), value :: arg
      integer :: c_pthread_create
    end function c_pthread_create
    function c_pthread_setaffinity_np(tid, size, mask) bind(c, name='pthread_setaffinity_np')
      import :: i8
      integer(i8), value :: tid
      integer(i8), value :: size
      integer, intent(in) :: mask(16)
      integer :: c_pthread_setaffinity_np
    end function c_pthread_setaffinity_np
    function c_mmap(addr, len, prot, flags, fd, off) bind(c, name='mmap')
      import :: i8
      type(c_ptr), value :: addr
      integer(i8), value :: len, fd, off
      integer, value :: prot, flags
      type(c_ptr) :: c_mmap
    end function c_mmap
    function c_mprotect(addr, len, prot) bind(c, name='mprotect')
      import :: i8
      type(c_ptr), value :: addr
      integer(i8), value :: len
      integer, value :: prot
      integer :: c_mprotect
    end function c_mprotect
    subroutine c_memcpy(dst, src, n) bind(c, name='memcpy')
      import :: i8
      type(c_ptr), value :: dst, src
      integer(i8), value :: n
    end subroutine c_memcpy
    subroutine c_usleep(us) bind(c, name='usleep')
      import :: i8
      integer(i8), value :: us
    end subroutine c_usleep
  end interface

contains

  subroutine nop_spin()
    g_spin = g_spin + 1
  end subroutine nop_spin

  function in_mask(cpu) result(r)
    integer, intent(in) :: cpu
    logical :: r
    r = (iand(mask32(cpu/32+1), ishft(1_4, mod(cpu,32))) /= 0)
  end function in_mask

  subroutine set_mask_bit(m, cpu)
    integer, intent(out) :: m(16)
    integer, intent(in) :: cpu
    m = 0
    m(cpu/32+1) = ishft(1_4, mod(cpu,32))
  end subroutine set_mask_bit

  subroutine parse_range(f, lo, hi)
    character(len=*), intent(in) :: f
    integer, intent(out) :: lo, hi
    integer :: p, io
    p = index(f, '-')
    if (p == 0) then
      read(f, *, iostat=io) lo
      hi = lo
    else
      read(f(1:p-1), *, iostat=io) lo
      read(f(p+1:), *, iostat=io) hi
    end if
    if (io /= 0) then
      lo = 1
      hi = 0
    end if
  end subroutine parse_range

  subroutine parse_cpulist(s, lst, cnt)
    character(len=*), intent(in) :: s
    integer, intent(out) :: lst(192)
    integer, intent(out) :: cnt
    character(len=256) :: t
    character(len=64) :: field
    integer :: p, lo, hi
    t = adjustl(s)
    cnt = 0
    do
      p = index(t, ',')
      if (p == 0) then
        field = t
        t = ''
      else
        field = t(1:p-1)
        t = adjustl(t(p+1:))
      end if
      call parse_range(field, lo, hi)
      do
        cnt = cnt + 1
        lst(cnt) = lo
        lo = lo + 1
        if (lo > hi) exit
      end do
      if (len_trim(t) == 0) exit
    end do
  end subroutine parse_cpulist

  subroutine read_cpulist(node, lst, cnt)
    integer, intent(in) :: node
    integer, intent(out) :: lst(192)
    integer, intent(out) :: cnt
    character(len=512) :: line
    character(len=128) :: f
    integer :: iu, io
    write(f, '(a,i0,a)') '/sys/devices/system/node/node', node, '/cpulist'
    cnt = 0
    open(newunit=iu, file=f, status='old', action='read', iostat=io)
    if (io /= 0) return
    read(iu, '(a)', iostat=io) line
    close(iu)
    if (io /= 0) return
    call parse_cpulist(line, lst, cnt)
  end subroutine read_cpulist

  function probe_node(cpu, a, n) result(tms)
    use omp_lib
    implicit none
    integer, intent(in) :: cpu
    real(kind=8), intent(in) :: a(*)
    integer(i8), intent(in) :: n
    real(kind=8) :: tms
    real(kind=8) :: acc
    real(kind=8) :: t0, t1
    integer :: pm(16), rc, p
    integer(i8) :: base
    integer(i8) :: stride
    integer(i8) :: cnt2
    call set_mask_bit(pm, cpu)
    rc = c_sched_setaffinity(0_8, 64_8, pm)
    if (rc /= 0) then
      tms = 1.0e9_8
      return
    end if
    t0 = omp_get_wtime()
    acc = 0.0_8
    if (n >= 1048576_8) then
      stride = n / 128_8
      do p = 1, 128
        base = (p-1_8)*stride + 1_8
        cnt2 = 0_8
        do
          acc = acc + a(base)
          base = base + 1_8
          cnt2 = cnt2 + 1_8
          if (cnt2 >= 8192_8 .or. base > n) exit
        end do
      end do
    else
      acc = a(1_8)
    end if
    t1 = omp_get_wtime()
    g_probe_acc = g_probe_acc + acc
    call c_sched_setaffinity(0_8, 64_8, mask32)
    tms = t1 - t0
  end function probe_node

  subroutine swap_int(x, y)
    integer, intent(inout) :: x, y
    integer :: z
    z = x
    x = y
    y = z
  end subroutine swap_int

  subroutine ensure_init(a, n)
    implicit none
    real(kind=8), intent(in) :: a(*)
    integer(i8), intent(in) :: n
    integer :: rc, cnt, k, i, t
    integer :: best
    integer(i8) :: tid_out, shv
    type(c_funptr) :: pfun
    integer :: nodecnt(0:3), nodecpus(0:3,192)
    integer(i8) :: ntnode(0:3)
    real(kind=8) :: bwnode(0:3)
    logical :: have_node(0:3)
    integer :: order(4)
    integer :: pm(16)
    character(len=16) :: envv

    if (g_ready .or. g_failed) return
    call get_command_environment('HPCAG_DEBUG', envv)
    if (trim(envv) /= '') g_dbg = .true.

    rc = c_sched_getaffinity(0_8, 64_8, mask32)
    if (rc /= 0) then
      g_failed = .true.
      return
    end if
    cnt = 0
    do i = 0, 191
      if (in_mask(i)) cnt = cnt + 1
    end do
    if (cnt < 4) then
      g_failed = .true.
      return
    end if
    do k = 0, 3
      have_node(k) = .false.
      bwnode(k) = 1.0e30_8
      ntnode(k) = -1_8
      call read_cpulist(k, nodecpus(k,:), nodecnt(k))
      do i = 1, nodecnt(k)
        if (in_mask(nodecpus(k,i))) then
          ntnode(k) = int(nodecpus(k,i), i8)
          exit
        end if
      end do
      if (ntnode(k) >= 0_8) have_node(k) = .true.
    end do
    if (count(have_node) == 0) then
      g_failed = .true.
      return
    end if
    do k = 0, 3
      if (have_node(k)) bwnode(k) = probe_node(int(ntnode(k), 4), a, n)
    end do
    order = [0,1,2,3]
    do i = 1, 3
      best = i
      do k = i+1, 4
        if (bwnode(order(k)) < bwnode(order(best))) best = k
      end do
      if (best /= i) call swap_int(order(i), order(best))
    end do
    g_nw = 0
    do i = 1, 4
      k = order(i)
      if (.not. have_node(k)) cycle
      do t = 1, nodecnt(k)
        if (g_nw >= ntw) exit
        if (.not. in_mask(nodecpus(k,t))) cycle
        g_nw = g_nw + 1
        pin_cpu(g_nw) = int(nodecpus(k,t), i8)
      end do
      if (g_nw >= ntw) exit
    end do
    if (g_nw < 2) then
      g_failed = .true.
      return
    end if
    blob_ptr = c_mmap(c_null_ptr, 4096_8, 3_4, 34_4, -1_8, 0_8)
    shv = transfer(blob_ptr, shv)
    if (shv == 0_8 .or. shv == -1_8) then
      g_failed = .true.
      return
    end if
    call c_memcpy(blob_ptr, c_loc(blob), blobsize)
    rc = c_mprotect(blob_ptr, 4096_8, 1_4)
    if (rc /= 0) then
      g_failed = .true.
      return
    end if
    shv = transfer(c_loc(g), shv)
    pfun = transfer(blob_ptr, pfun)
    do t = 0_8, int(g_nw-1, i8)
      call c_pthread_create(tid_out, c_null_ptr, pfun, shv + 16_8 + int(t, i8)*64_8)
      call set_mask_bit(pm, int(pin_cpu(t+1), 4))
      call c_pthread_setaffinity_np(tid_out, 64_8, pm)
    end do
    g%seq = 0_8
    g%done = 0_8
    g_ready = .true.
    if (g_dbg) write(*, '(a,i3)') ' [hpcag] workers=', g_nw
  end subroutine ensure_init

  subroutine run_fast(pa, pb, n, sd)
    implicit none
    integer(i8), intent(in) :: pa, pb, n
    real(kind=8), intent(in) :: sd
    integer(i8) :: shv, t, hi, per, rem, c0
    integer :: i, waitc
    shv = transfer(c_loc(g%seq), shv)
    per = n / int(g_nw, i8)
    rem = mod(n, int(g_nw, i8))
    c0 = 0_8
    do t = 0_8, int(g_nw-1, i8)
      hi = c0 + per
      if (t < rem) hi = c0 + per + 1_8
      g%sl(t)%a = pa
      g%sl(t)%b = pb
      g%sl(t)%lo = c0
      g%sl(t)%hi = hi
      g%sl(t)%sd = sd
      g%sl(t)%tid = t
      g%sl(t)%sh = shv
      c0 = hi
    end do
    g%done = 0_8
    g%seq = g%seq + 1_8
    waitc = 0
    do
      if (g%done >= int(g_nw, i8)) exit
      do i = 1, 2000
        call nop_spin
      end do
      call c_usleep(1)
      waitc = waitc + 1
      if (waitc > 30000 .and. g_wt_printed == 0) then
        g_wt_printed = 1
        write(*, '(a)') ' [hpcag] worker wait exceeded 30s'
      end if
    end do
  end subroutine run_fast

  subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(C, name='tsvc_2_vpvts_fp64')
    use iso_c_binding
    implicit none
    real(kind=c_double) :: a(*)
    real(kind=c_double) :: b(*)
    integer(kind=c_int64_t), value :: len_1d
    integer(kind=c_int64_t), value :: s
    integer(i8) :: i
    real(kind=c_double) :: sd
    integer(i8) :: pa, pb

    if (len_1d <= 0_8) return
    sd = real(s, kind=8)
    if (len_1d >= 8000000_8) then
      if (.not. g_ready .and. .not. g_failed) call ensure_init(a, len_1d)
      if (g_ready) then
        pa = transfer(c_loc(a(1)), pa)
        pb = transfer(c_loc(b(1)), pb)
        call run_fast(pa, pb, len_1d, sd)
        return
      end if
    end if
    if (len_1d < 16384_8) then
      do i = 1_8, len_1d
        a(i) = a(i) + b(i) * sd
      end do
    else
      !$omp parallel do schedule(static)
      do i = 1_8, len_1d
        a(i) = a(i) + b(i) * sd
      end do
      !$omp end parallel do
    end if
  end subroutine tsvc_2_vpvts_fp64


end module kernmod
