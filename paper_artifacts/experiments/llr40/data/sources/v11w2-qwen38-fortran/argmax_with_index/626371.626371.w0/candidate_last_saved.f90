module amax_mod
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  private
  public :: argmax_with_index_fp64

  real(c_double), allocatable :: pv(:)
  integer(c_int64_t), allocatable :: pp(:)
  integer, save :: pcap = 0

contains

  subroutine ensure_cap(c)
    implicit none
    integer, intent(in) :: c
    if (c > pcap) then
      if (pcap > 0) deallocate(pv, pp)
      allocate(pv(c), pp(c))
      pcap = c
    end if
  end subroutine ensure_cap

  subroutine scan_range(a, lo, hi, mv, mp)
    implicit none
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(in) :: lo, hi
    real(c_double), intent(out) :: mv
    integer(c_int64_t), intent(out) :: mp
    integer(c_int64_t) :: i, ii, nb
    real(c_double) :: x, v1(8), v2(8), v3(8), v4(8), m1, m2, m3, m4
    integer(c_int64_t) :: p

    x = -huge(0.0d0)
    p = huge(0_8)
    nb = (hi - lo) / 32_8
    do ii = 0_8, nb - 1_8
      i = lo + 32_8*ii
      v1 = a(i:i+7)
      v2 = a(i+8:i+15)
      v3 = a(i+16:i+23)
      v4 = a(i+24:i+31)
      m1 = max(x, v1(1), v1(2), v1(3), v1(4), v1(5), v1(6), v1(7), v1(8))
      m2 = max(x, v2(1), v2(2), v2(3), v2(4), v2(5), v2(6), v2(7), v2(8))
      m3 = max(x, v3(1), v3(2), v3(3), v3(4), v3(5), v3(6), v3(7), v3(8))
      m4 = max(x, v4(1), v4(2), v4(3), v4(4), v4(5), v4(6), v4(7), v4(8))
      if (m1 > x) then
        x = m1
        if (v1(1) == x) then; p = i
        else if (v1(2) == x) then; p = i + 1_8
        else if (v1(3) == x) then; p = i + 2_8
        else if (v1(4) == x) then; p = i + 3_8
        else if (v1(5) == x) then; p = i + 4_8
        else if (v1(6) == x) then; p = i + 5_8
        else if (v1(7) == x) then; p = i + 6_8
        else; p = i + 7_8
        end if
      end if
      if (m2 > x) then
        x = m2
        if (v2(1) == x) then; p = i + 8_8
        else if (v2(2) == x) then; p = i + 9_8
        else if (v2(3) == x) then; p = i + 10_8
        else if (v2(4) == x) then; p = i + 11_8
        else if (v2(5) == x) then; p = i + 12_8
        else if (v2(6) == x) then; p = i + 13_8
        else if (v2(7) == x) then; p = i + 14_8
        else; p = i + 15_8
        end if
      end if
      if (m3 > x) then
        x = m3
        if (v3(1) == x) then; p = i + 16_8
        else if (v3(2) == x) then; p = i + 17_8
        else if (v3(3) == x) then; p = i + 18_8
        else if (v3(4) == x) then; p = i + 19_8
        else if (v3(5) == x) then; p = i + 20_8
        else if (v3(6) == x) then; p = i + 21_8
        else if (v3(7) == x) then; p = i + 22_8
        else; p = i + 23_8
        end if
      end if
      if (m4 > x) then
        x = m4
        if (v4(1) == x) then; p = i + 24_8
        else if (v4(2) == x) then; p = i + 25_8
        else if (v4(3) == x) then; p = i + 26_8
        else if (v4(4) == x) then; p = i + 27_8
        else if (v4(5) == x) then; p = i + 28_8
        else if (v4(6) == x) then; p = i + 29_8
        else if (v4(7) == x) then; p = i + 30_8
        else; p = i + 31_8
        end if
      end if
    end do
    do i = lo + 32_8*nb, hi
      if (a(i) > x) then
        x = a(i)
        p = i
      end if
    end do
    mv = x
    mp = p
  end subroutine scan_range

  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, wsptr, wssize) &
       bind(C, name='argmax_with_index_fp64')
    implicit none
    real(c_double),            intent(in)  :: a(*)
    integer(c_int64_t),        intent(out) :: out_index
    real(c_double),            intent(out) :: out_value
    integer(c_int64_t), value, intent(in)  :: LEN_1D
    type(c_ptr),               intent(in)  :: wsptr
    integer(c_int64_t), value, intent(in)  :: wssize

    integer(c_int64_t) :: n, lo, hi, chunk
    real(c_double) :: x
    integer(c_int64_t) :: p
    integer :: nt, t

    n = LEN_1D
    if (n <= 0_8) then
      out_value = 0.0d0
      out_index = 1_8
      return
    end if
    if (a(1) /= a(1)) then
      out_value = a(1)
      out_index = 1_8
      return
    end if
    if (n <= 8388608_8) then
      call scan_range(a, 1_8, n, x, p)
      out_value = x
      out_index = p
      return
    end if

    nt = omp_get_max_threads()
    if (nt < 1) nt = 1
    if (nt > 256) nt = 256
    if (int(nt, 8)*32_8 > n) nt = max(1, int(n/32_8, 4))
    chunk = (n + int(nt, 8) - 1_8) / int(nt, 8)
    call ensure_cap(nt)
  !$omp parallel do num_threads(nt) schedule(static)
    do t = 0, nt - 1
      lo = int(t, 8)*chunk + 1_8
      hi = min(n, int(t + 1_8, 8)*chunk)
      if (lo <= hi) then
        call scan_range(a, lo, hi, pv(t + 1), pp(t + 1))
      else
        pv(t + 1) = -huge(0.0d0)
        pp(t + 1) = huge(0_8)
      end if
    end do
  !$omp end parallel do
    x = pv(1)
    p = pp(1)
    do t = 2, nt
      if (pv(t) > x) then
        x = pv(t)
        p = pp(t)
      end if
    end do
    out_value = x
    out_index = p
  end subroutine argmax_with_index_fp64
end module amax_mod
