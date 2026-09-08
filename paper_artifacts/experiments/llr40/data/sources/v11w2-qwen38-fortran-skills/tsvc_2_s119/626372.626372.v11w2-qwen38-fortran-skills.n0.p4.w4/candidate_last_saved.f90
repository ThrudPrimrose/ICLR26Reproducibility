integer(c_int64_t) function rdflag(v, k)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in) :: v(:)
  integer(c_int64_t), intent(in) :: k
  rdflag = v(k)
end function rdflag

subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), parameter :: c64 = 8
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: nt, M, W, base, rem, t, s, e, wf, i, j
  integer(c_int64_t), allocatable :: rd(:)
  real(c_double), allocatable, target :: b0(:), b1(:)
  real(c_double), pointer :: cur(:), nxt(:), ptmp(:)
  interface
    integer(8) function rdflag(v, k)
      integer(8), intent(in) :: v(:)
      integer(8), intent(in) :: k
    end function
  end interface

  if (LEN_2D < 2_c64) return
  nt = omp_get_max_threads()
  M = LEN_2D - 1_c64

  if (M < 16_c64*nt) then
    !$omp parallel shared(aa, bb, LEN_2D) private(i, j)
    do i = 2_c64, LEN_2D
      !$omp do simd
      do j = 2_c64, LEN_2D
        aa(j, i) = aa(j-1_c64, i-1_c64) + bb(j, i)
      end do
    end do
    !$omp end parallel
    return
  end if

  allocate(rd(0_c64:16_c64*nt))
  rd(0_c64:) = 1_c64
  base = M / nt
  rem = M - base*nt

  !$omp parallel shared(aa, bb, LEN_2D, M, nt, base, rem, rd) private(t, s, e, wf, W, i, j, b0, b1, cur, nxt, ptmp)
  t = omp_get_thread_num()
  W = base + merge(1_c64, 0_c64, t < rem)
  s = 2_c64 + t*base + min(t, rem)
  e = s + W - 1_c64
  allocate(b0(W), b1(W))
  cur => b0
  nxt => b1
  !$omp simd
  do j = s, e
    cur(j - s + 1_c64) = bb(j, 2_c64)
  end do
  wf = t - 1_c64
  do i = 2_c64, LEN_2D
    if (t > 0_c64) then
      do
        if (rdflag(rd, wf*16_c64) >= i - 1_c64) exit
      end do
    end if
    if (i < LEN_2D) then
      !$omp simd
      do j = s, e
        nxt(j - s + 1_c64) = bb(j, i + 1_c64)
      end do
    end if
    !$omp simd
    do j = s, e
      aa(j, i) = aa(j-1_c64, i-1_c64) + cur(j - s + 1_c64)
    end do
    ptmp => cur
    cur => nxt
    nxt => ptmp
    rd(t*16_c64) = i
  end do
  !$omp end parallel
  deallocate(rd)
end subroutine tsvc_2_s119_fp64
