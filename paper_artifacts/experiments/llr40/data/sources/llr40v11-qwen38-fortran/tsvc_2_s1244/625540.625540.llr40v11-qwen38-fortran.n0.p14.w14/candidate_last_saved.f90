! TSVC tsvc_2 kernel s1244 (1-based, N = LEN_1D):
!   a(i) = g(i)                    for i = 1..N-1
!   d(i) = g(i) + a(i+1)_OLD       for i = 1..N-1
! where g(i) = b(i) + c(i)^2 + b(i)^2 + c(i).  a(N) is never written.
! In the sequential oracle d(i) sees the ORIGINAL a(i+1) (its own update
! lands one iteration later), so each thread must capture a(hi+1) (its right
! boundary) before any thread writes; one barrier separates the two phases.
! Traffic: b,c once (16N) + a read (8N) + d (8N) + a write (8N) = 40 B/elt.
subroutine tsvc_2_s1244_fp64(pa, pb, pc, pd, len_1d) bind(C, name='tsvc_2_s1244_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: pa, pb, pc, pd
  integer(c_int64_t), value :: len_1d
  real(c_double), dimension(:), pointer :: a, b, c, d
  integer(c_int64_t) :: n, i, lo, hi, chunk
  integer :: nthr
  real(c_double) :: s, g

  n = len_1d - 1
  if (n < 1) return

  call c_f_pointer(pa, a, [len_1d])
  call c_f_pointer(pb, b, [len_1d])
  call c_f_pointer(pc, c, [len_1d])
  call c_f_pointer(pd, d, [len_1d])

  nthr = omp_get_max_threads()
  if (nthr < 1) nthr = 1
  !$omp parallel default(none) shared(a,b,c,n,nthr)
  block
    integer(c_int64_t) :: tid
    real(c_double) :: s, g
    integer(c_int64_t) :: i, lo, hi, chunk
    tid = omp_get_thread_num()
    chunk = (n + nthr - 1) / nthr
    lo = tid * chunk + 1
    hi = n
    if (lo + chunk - 1 < n) hi = lo + chunk - 1
    if (lo <= hi) then
      s = a(hi + 1)      ! capture original right-boundary value
      !$omp barrier
      do i = lo, hi - 1
        g = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
        d(i) = g + a(i+1)
        a(i) = g
      end do
      g = b(hi) + c(hi)*c(hi) + b(hi)*b(hi) + c(hi)
      d(hi) = g + s
      a(hi) = g
    else
      !$omp barrier
    end if
  end block
  !$omp end parallel
end subroutine tsvc_2_s1244_fp64
