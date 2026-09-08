! TSVC tsvc_2 kernel s252 (fp64) -- C-ABI Fortran, optimized.
!
! Reference (serial):
!   t = 0
!   for i: s = b[i]*c[i]; a[i] = s + t; t = s
! Since t only ever holds the PREVIOUS s, this reduces to
!   a[i] = d[i] + d[i-1],   d[i] = b[i]*c[i],   a[0] = d[0]
! i.e. a one-element shifted add of the elementwise product. There is no true
! loop-carried dependency, so we split it into two trivially-vectorizable
! passes over a scratch array d and parallelize by manual partitioning.
!
! gfortran will not auto-vectorize a loop that lives inside an OpenMP
! construct, so the (vectorized) passes live in plain subroutines that the
! OpenMP region calls once per thread on that thread's contiguous chunk.
subroutine tsvc_2_s252_fp64(ap, bp, cp, len_1d, wsp, wsb) bind(c, name="tsvc_2_s252_fp64")
  use, intrinsic :: iso_c_binding, only: c_ptr, c_double, c_int64_t, c_f_pointer
  use, intrinsic :: omp_lib
  implicit none
  type(c_ptr),          value :: ap, bp, cp, wsp
  integer(c_int64_t),   value :: len_1d
  integer(c_int64_t),   value :: wsb
  real(c_double), contiguous, dimension(:), pointer :: fa, fb, fc, fd
  integer :: n, nt, rank, lo, hi, size, chunk

  interface
    subroutine tsvc2_pass1(b, c, d, lo, hi)
      real(8), contiguous, dimension(:), intent(in)  :: b, c
      real(8), contiguous, dimension(:), intent(out) :: d
      integer, intent(in) :: lo, hi
    end subroutine tsvc2_pass1
    subroutine tsvc2_pass2(d, a, lo, hi)
      real(8), contiguous, dimension(:), intent(in)  :: d
      real(8), contiguous, dimension(:), intent(out) :: a
      integer, intent(in) :: lo, hi
    end subroutine tsvc2_pass2
  end interface

  n = int(len_1d)
  if (n <= 1) then
    if (n == 1) then
      call c_f_pointer(ap, fa, [1])
      call c_f_pointer(bp, fb, [1])
      call c_f_pointer(cp, fc, [1])
      fa(1) = fb(1) * fc(1)
    end if
    return
  end if

  call c_f_pointer(ap, fa, [n])
  call c_f_pointer(bp, fb, [n])
  call c_f_pointer(cp, fc, [n])
  if (wsb >= int(8, c_int64_t) * int(n, c_int64_t)) then
    call c_f_pointer(wsp, fd, [n])
  else
    allocate(fd(n))
  end if

  if (n < 131072) then
    ! small: run serial, no team-spawn overhead
    call tsvc2_pass1(fb, fc, fd, 1, n)
    call tsvc2_pass2(fd, fa, 2, n)
    fa(1) = fd(1)
  else
    !$omp parallel private(nt, rank, lo, hi, size, chunk)
      nt = omp_get_num_threads()
      rank = omp_get_thread_num()
      ! pass 1: fd(i) = fb(i)*fc(i)   over [1, n]
      size = n
      chunk = (size + nt - 1) / nt
      lo = 1 + rank * chunk
      hi = min(n, 1 + (rank + 1) * chunk - 1)
      if (lo <= hi) call tsvc2_pass1(fb, fc, fd, lo, hi)
      !$omp barrier
      ! pass 2: fa(i) = fd(i)+fd(i-1)   over [2, n]
      size = n - 1
      chunk = (size + nt - 1) / nt
      lo = 2 + rank * chunk
      hi = min(n, 2 + (rank + 1) * chunk - 1)
      if (lo <= hi) call tsvc2_pass2(fd, fa, lo, hi)
    !$omp end parallel
    fa(1) = fd(1)
  end if

  if (wsb < int(8, c_int64_t) * int(n, c_int64_t)) deallocate(fd)
end subroutine tsvc_2_s252_fp64

! d(lo:hi) = b * c   -- vectorizes to 64-byte (AVX-512) with unroll 8
subroutine tsvc2_pass1(b, c, d, lo, hi)
  real(8), contiguous, dimension(:), intent(in)  :: b, c
  real(8), contiguous, dimension(:), intent(out) :: d
  integer, intent(in) :: lo, hi
  integer :: i
  do i = lo, hi
    d(i) = b(i) * c(i)
  end do
end subroutine tsvc2_pass1

! a(lo:hi) = d(i) + d(i-1)   -- vectorizes to 64-byte (AVX-512) with unroll 8
subroutine tsvc2_pass2(d, a, lo, hi)
  real(8), contiguous, dimension(:), intent(in)  :: d
  real(8), contiguous, dimension(:), intent(out) :: a
  integer, intent(in) :: lo, hi
  integer :: i
  do i = lo, hi
    a(i) = d(i) + d(i - 1)
  end do
end subroutine tsvc2_pass2
