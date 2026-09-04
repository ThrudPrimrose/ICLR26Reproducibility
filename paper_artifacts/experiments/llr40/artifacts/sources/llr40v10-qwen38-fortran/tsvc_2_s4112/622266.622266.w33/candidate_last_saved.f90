! TSVC_2 s4112: a(i) = a(i) + b(ip(i)) * 2.0
! OpenMP block-partitioned; the inner loop lives in an internal subroutine so
! gfortran's vectorizer can handle it (OMP shared vars are double-indirected
! inside the region fn, which blocks vectorization).
subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(c, name="tsvc_2_s4112_fp64")
  use, intrinsic :: omp_lib
  implicit none
  real(kind=8), intent(inout) :: a(*)
  real(kind=8), intent(in)    :: b(*)
  integer(kind=4), intent(in) :: ip(*)
  integer(kind=8), value      :: LEN_1D
  integer(kind=8) :: n, lo, hi, nt, tid, chunk
  integer :: tnum

  n = LEN_1D
  if (n < 131072) then
    call tsvc_kernel(a, b, ip, int(1, 8), n)
  else
    tnum = 4
    !$omp parallel default(none) num_threads(tnum) shared(a,b,ip,n) private(lo,hi,nt,tid,chunk)
      tid   = omp_get_thread_num()
      nt    = omp_get_num_threads()
      chunk = (n + nt - 1) / nt
      lo    = tid * chunk + 1
      hi    = min(lo + chunk - 1, n)
      if (lo <= hi) call tsvc_kernel(a, b, ip, lo, hi)
    !$omp end parallel
  end if

contains

  subroutine tsvc_kernel(aa, bb, ip_, lo, hi)
    implicit none
    real(kind=8), intent(inout) :: aa(*)
    real(kind=8), intent(in)    :: bb(*)
    integer(kind=4), intent(in) :: ip_(*)
    integer(kind=8), value      :: lo, hi
    integer(kind=8) :: i

    do i = lo, hi
      aa(i) = aa(i) + bb(ip_(i)) * 2.0d0
    end do
  end subroutine tsvc_kernel
end subroutine tsvc_2_s4112_fp64
