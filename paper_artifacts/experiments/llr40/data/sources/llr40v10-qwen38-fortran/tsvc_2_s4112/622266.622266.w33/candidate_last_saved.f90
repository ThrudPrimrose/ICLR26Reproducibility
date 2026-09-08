! TSVC_2 s4112: a(i) = a(i) + b(ip(i)) * 2.0
! 24-thread OpenMP block partitioning; 64-elem unrolled body per thread
! (8 independent 8-elem vector batches) to maximize memory-level parallelism.
subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(c, name="tsvc_2_s4112_fp64")
  use, intrinsic :: omp_lib
  implicit none
  real(kind=8), intent(inout) :: a(*)
  real(kind=8), intent(in)    :: b(*)
  integer(kind=4), intent(in) :: ip(*)
  integer(kind=8), value      :: LEN_1D
  integer(kind=8) :: n, lo, hi, nt, tid, chunk

  n = LEN_1D
  if (n < 262144) then
    call tsvc_kernel(a, b, ip, int(1, 8), n)
  else
    !$omp parallel default(none) shared(a,b,ip,n) private(lo,hi,nt,tid,chunk)
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
    integer(kind=8) :: i0, i2, j

    i2 = lo
    do
      if (i2 + 63 > hi) exit
      aa(i0 + 0) = aa(i0 + 0) + bb(ip_(i0 + 0)) * 2.0d0
      aa(i0 + 1) = aa(i0 + 1) + bb(ip_(i0 + 1)) * 2.0d0
      aa(i0 + 2) = aa(i0 + 2) + bb(ip_(i0 + 2)) * 2.0d0
      aa(i0 + 3) = aa(i0 + 3) + bb(ip_(i0 + 3)) * 2.0d0
      aa(i0 + 4) = aa(i0 + 4) + bb(ip_(i0 + 4)) * 2.0d0
      aa(i0 + 5) = aa(i0 + 5) + bb(ip_(i0 + 5)) * 2.0d0
      aa(i0 + 6) = aa(i0 + 6) + bb(ip_(i0 + 6)) * 2.0d0
      aa(i0 + 7) = aa(i0 + 7) + bb(ip_(i0 + 7)) * 2.0d0
      aa(i0 + 8) = aa(i0 + 8) + bb(ip_(i0 + 8)) * 2.0d0
      aa(i0 + 9) = aa(i0 + 9) + bb(ip_(i0 + 9)) * 2.0d0
      aa(i0 + 10) = aa(i0 + 10) + bb(ip_(i0 + 10)) * 2.0d0
      aa(i0 + 11) = aa(i0 + 11) + bb(ip_(i0 + 11)) * 2.0d0
      aa(i0 + 12) = aa(i0 + 12) + bb(ip_(i0 + 12)) * 2.0d0
      aa(i0 + 13) = aa(i0 + 13) + bb(ip_(i0 + 13)) * 2.0d0
      aa(i0 + 14) = aa(i0 + 14) + bb(ip_(i0 + 14)) * 2.0d0
      aa(i0 + 15) = aa(i0 + 15) + bb(ip_(i0 + 15)) * 2.0d0
      aa(i0 + 16) = aa(i0 + 16) + bb(ip_(i0 + 16)) * 2.0d0
      aa(i0 + 17) = aa(i0 + 17) + bb(ip_(i0 + 17)) * 2.0d0
      aa(i0 + 18) = aa(i0 + 18) + bb(ip_(i0 + 18)) * 2.0d0
      aa(i0 + 19) = aa(i0 + 19) + bb(ip_(i0 + 19)) * 2.0d0
      aa(i0 + 20) = aa(i0 + 20) + bb(ip_(i0 + 20)) * 2.0d0
      aa(i0 + 21) = aa(i0 + 21) + bb(ip_(i0 + 21)) * 2.0d0
      aa(i0 + 22) = aa(i0 + 22) + bb(ip_(i0 + 22)) * 2.0d0
      aa(i0 + 23) = aa(i0 + 23) + bb(ip_(i0 + 23)) * 2.0d0
      aa(i0 + 24) = aa(i0 + 24) + bb(ip_(i0 + 24)) * 2.0d0
      aa(i0 + 25) = aa(i0 + 25) + bb(ip_(i0 + 25)) * 2.0d0
      aa(i0 + 26) = aa(i0 + 26) + bb(ip_(i0 + 26)) * 2.0d0
      aa(i0 + 27) = aa(i0 + 27) + bb(ip_(i0 + 27)) * 2.0d0
      aa(i0 + 28) = aa(i0 + 28) + bb(ip_(i0 + 28)) * 2.0d0
      aa(i0 + 29) = aa(i0 + 29) + bb(ip_(i0 + 29)) * 2.0d0
      aa(i0 + 30) = aa(i0 + 30) + bb(ip_(i0 + 30)) * 2.0d0
      aa(i0 + 31) = aa(i0 + 31) + bb(ip_(i0 + 31)) * 2.0d0
      aa(i0 + 32) = aa(i0 + 32) + bb(ip_(i0 + 32)) * 2.0d0
      aa(i0 + 33) = aa(i0 + 33) + bb(ip_(i0 + 33)) * 2.0d0
      aa(i0 + 34) = aa(i0 + 34) + bb(ip_(i0 + 34)) * 2.0d0
      aa(i0 + 35) = aa(i0 + 35) + bb(ip_(i0 + 35)) * 2.0d0
      aa(i0 + 36) = aa(i0 + 36) + bb(ip_(i0 + 36)) * 2.0d0
      aa(i0 + 37) = aa(i0 + 37) + bb(ip_(i0 + 37)) * 2.0d0
      aa(i0 + 38) = aa(i0 + 38) + bb(ip_(i0 + 38)) * 2.0d0
      aa(i0 + 39) = aa(i0 + 39) + bb(ip_(i0 + 39)) * 2.0d0
      aa(i0 + 40) = aa(i0 + 40) + bb(ip_(i0 + 40)) * 2.0d0
      aa(i0 + 41) = aa(i0 + 41) + bb(ip_(i0 + 41)) * 2.0d0
      aa(i0 + 42) = aa(i0 + 42) + bb(ip_(i0 + 42)) * 2.0d0
      aa(i0 + 43) = aa(i0 + 43) + bb(ip_(i0 + 43)) * 2.0d0
      aa(i0 + 44) = aa(i0 + 44) + bb(ip_(i0 + 44)) * 2.0d0
      aa(i0 + 45) = aa(i0 + 45) + bb(ip_(i0 + 45)) * 2.0d0
      aa(i0 + 46) = aa(i0 + 46) + bb(ip_(i0 + 46)) * 2.0d0
      aa(i0 + 47) = aa(i0 + 47) + bb(ip_(i0 + 47)) * 2.0d0
      aa(i0 + 48) = aa(i0 + 48) + bb(ip_(i0 + 48)) * 2.0d0
      aa(i0 + 49) = aa(i0 + 49) + bb(ip_(i0 + 49)) * 2.0d0
      aa(i0 + 50) = aa(i0 + 50) + bb(ip_(i0 + 50)) * 2.0d0
      aa(i0 + 51) = aa(i0 + 51) + bb(ip_(i0 + 51)) * 2.0d0
      aa(i0 + 52) = aa(i0 + 52) + bb(ip_(i0 + 52)) * 2.0d0
      aa(i0 + 53) = aa(i0 + 53) + bb(ip_(i0 + 53)) * 2.0d0
      aa(i0 + 54) = aa(i0 + 54) + bb(ip_(i0 + 54)) * 2.0d0
      aa(i0 + 55) = aa(i0 + 55) + bb(ip_(i0 + 55)) * 2.0d0
      aa(i0 + 56) = aa(i0 + 56) + bb(ip_(i0 + 56)) * 2.0d0
      aa(i0 + 57) = aa(i0 + 57) + bb(ip_(i0 + 57)) * 2.0d0
      aa(i0 + 58) = aa(i0 + 58) + bb(ip_(i0 + 58)) * 2.0d0
      aa(i0 + 59) = aa(i0 + 59) + bb(ip_(i0 + 59)) * 2.0d0
      aa(i0 + 60) = aa(i0 + 60) + bb(ip_(i0 + 60)) * 2.0d0
      aa(i0 + 61) = aa(i0 + 61) + bb(ip_(i0 + 61)) * 2.0d0
      aa(i0 + 62) = aa(i0 + 62) + bb(ip_(i0 + 62)) * 2.0d0
      aa(i0 + 63) = aa(i0 + 63) + bb(ip_(i0 + 63)) * 2.0d0
      i2 = i2 + 64
    end do
    do j = i2, hi
      aa(j) = aa(j) + bb(ip_(j)) * 2.0d0
    end do
  end subroutine tsvc_kernel
end subroutine tsvc_2_s4112_fp64
