subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d, workspace, workspace_size) bind(C)
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  integer, parameter :: ik = c_int64_t
  integer(ik), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int32_t), intent(in) :: ip(len_1d)
  real(c_double), intent(inout) :: workspace(*)
  integer(ik), value, intent(in) :: workspace_size

  interface
    function c_getpid() bind(C, name='getpid')
      use iso_c_binding
      integer(c_int) :: c_getpid
    end function c_getpid
    function c_sched_setaffinity(pid, csize, mask) bind(C, name='sched_setaffinity')
      use iso_c_binding
      integer(c_int), value :: pid
      integer(c_size_t), value :: csize
      type(c_ptr) :: mask
      integer(c_int) :: c_sched_setaffinity
    end function c_sched_setaffinity
    function c_getcpu(cpu, node) bind(C, name='getcpu')
      use iso_c_binding
      integer(c_int), intent(out) :: cpu, node
      integer(c_int) :: c_getcpu
    end function c_getcpu
  end interface

  integer(c_int) :: mycpu, mynode, sret
  integer(ik) :: base, nt, t, lo, hi, i, c
  integer(c_int32_t) :: ci, bi, ri
  integer(c_int32_t), target :: mask(32)
  integer(c_int32_t), parameter :: p2(8) = [1_4, 2_4, 4_4, 8_4, 16_4, 32_4, 64_4, 128_4]
  type(c_ptr) :: maskp

  if (len_1d <= 0) return

  mycpu = 0
  mynode = 0
  sret = c_getcpu(mycpu, mynode)
  base = 0_ik
  if (sret == 0) base = int(mynode, ik) * 48_ik

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  if (nt > len_1d) nt = len_1d

  !$omp parallel default(none) shared(a, b, ip, len_1d, nt, base) private(t, lo, hi, i, c, ci, bi, ri, mask, sret, maskp)
    t = omp_get_thread_num()
    if (base > 0_ik) then
      c = base + t
      if (c < 192_ik) then
        mask = 0_c_int32_t
        ci = int(c, 4)
        bi = ci / 8
        ri = ci - bi * 8_c_int32_t
        mask(bi + 1) = p2(ri + 1)
        maskp = c_loc(mask)
        sret = c_sched_setaffinity(c_getpid(), 256_c_size_t, maskp)
      end if
    end if
    lo = (len_1d * t) / nt
    hi = (len_1d * (t + 1_ik)) / nt
    !$omp simd
    do i = lo + 1_ik, hi
      a(i) = a(i) + 2.0d0 * b(ip(i))
    end do
  !$omp end parallel
end subroutine tsvc_2_s4112_fp64
