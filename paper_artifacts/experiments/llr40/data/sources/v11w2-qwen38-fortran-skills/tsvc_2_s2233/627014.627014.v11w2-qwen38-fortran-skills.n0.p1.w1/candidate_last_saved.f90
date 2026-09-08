subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C)
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int64_t) :: n, ng, g, t, k, i0, len, j2, mword, mbit
  real(c_double) :: s1(256), s2(256)
  integer(c_int) :: iu, ierr
  character(len=64) :: line, tok
  character(len=1024) :: statline
  character(len=2048) :: cline
  integer(c_int64_t) :: a, b, p, mcpu
  integer :: ipos, i
  logical :: have_mask, inrange
  integer(c_int64_t) :: mask(16)
  type(c_ptr) :: mp
  interface
    subroutine sched_setaffinity(pid, cpusetsize, m) bind(C)
      import :: c_int, c_int64_t, c_ptr
      integer(c_int), value :: pid
      integer(c_int64_t), value :: cpusetsize
      type(c_ptr), value :: m
    end subroutine sched_setaffinity
  end interface

  n = len_2d
  if (n < 9) return

  ! ---- NUMA: the benchmark data was first touched by the calling
  ! (main) thread; find its node and keep every worker thread on it ----
  mcpu = -1_8
  have_mask = .false.
  open(newunit=iu, file='/proc/self/stat', status='old', action='read', &
       iostat=ierr)
  if (ierr == 0) then
    read(iu, '(a1024)', iostat=ierr) statline
    close(iu)
    if (ierr == 0) call parse_cpu(statline, mcpu)
  end if

  if (mcpu >= 0_8) then
    ! walk the /sys NUMA nodes until one claims mcpu
    do i = 0, 63
      write(line, '(a,i0)') '/sys/devices/system/node/node', i
      open(newunit=iu, file=trim(line), status='old', action='read', &
           iostat=ierr)
      if (ierr /= 0) exit
      read(iu, '(a2048)', iostat=ierr) cline
      close(iu)
      mask = 0_8
      inrange = .false.
      if (ierr == 0 .and. len_trim(cline) > 0) then
        cline = cline(1:len_trim(cline)) // ','
        p = 1_8
        do
          ipos = index(cline(p:), ',')
          if (ipos == 0) exit
          tok = cline(p:p + ipos - 2)
          read(tok, *, iostat=ierr) a, b
          if (ierr /= 0) b = a
          do k = a, b
            if (k >= 0_8 .and. k < 1024_8) then
              mword = k / 64_8 + 1
              mbit = 1_8
              do j2 = 1, mod(k, 64_8)
                mbit = 2_8 * mbit
              end do
              mask(mword) = ior(mask(mword), mbit)
            end if
          end do
          p = p + ipos
        end do
        inrange = .true.
      end if
      if (inrange .and. mcpu < 1024_8) then
        mword = mcpu / 64_8 + 1
        mbit = 1_8
        do j2 = 1, mod(mcpu, 64_8)
          mbit = 2_8 * mbit
        end do
        if (ieor(mask(mword), mbit) /= 0_8) then
          have_mask = .true.
          exit
        end if
      end if
    end do
  end if
  mp = transfer(mask(1), mp)

  ! ---- main work ----
  ! numpy recurrences in Fortran (column-major axis reversal):
  !  part 1: aa(i,t) = aa(i,t-1) + cc(i,t), i = 9..n, chain over t
  !  part 2: bb(j,t) = bb(j,t-1) + cc(j,t), j = 9..n, chain over t
  ! Rows are grouped in blocks of 128 (two memory lines of a column); all
  ! 128 chains live in vector registers.  Both parts are stepped together
  ! at the same t, so their cc reads share the same cache lines.
  !$omp parallel num_threads(min(omp_get_max_threads(), 48)) default(none) shared(aa, bb, cc, n, have_mask, mp) private(ng, g, t, k, i0, len, s1, s2)
  if (have_mask) call sched_setaffinity(0, 128_8, mp)
  ng = (n - 8 + 127) / 128
  !$omp do schedule(static)
  do g = 0, ng - 1
    i0 = 9 + 128 * g
    len = min(128_8, n - 8 - 128 * g)
    do k = 1, len
      s1(k) = aa(i0 + k - 1, 8)
      s2(k) = bb(i0 + k - 1, 8)
    end do
    do t = 9, n
      s1(1:len) = s1(1:len) + cc(i0:i0 + len - 1, t)
      aa(i0:i0 + len - 1, t) = s1(1:len)
      s2(1:len) = s2(1:len) + cc(i0:i0 + len - 1, t)
      bb(i0:i0 + len - 1, t) = s2(1:len)
    end do
  end do
  !$omp end do
  !$omp end parallel
contains
  subroutine parse_cpu(s, outcpu)
    character(len=*), intent(in) :: s
    integer(c_int64_t), intent(out) :: outcpu
    integer :: ip, jt
    character(len=64) :: tt
    outcpu = -1_8
    ip = 0
    do jt = len_trim(s), 1, -1
      if (s(jt:jt) == ')') then
        ip = jt
        exit
      end if
    end do
    if (ip < 2) return
    ! after the last ')' come fields 3..39; field 39 is the processor
    ip = ip + 2
    do jt = 1, 37
      tt = ''
      do
        if (ip > len_trim(s)) return
        if (s(ip:ip) == ' ') then
          ip = ip + 1
          exit
        end if
        tt = tt // s(ip:ip)
        ip = ip + 1
      end do
      if (jt == 37) read(tt, *, iostat=ierr) outcpu
    end do
  end subroutine parse_cpu
end subroutine tsvc_2_s2233_fp64
