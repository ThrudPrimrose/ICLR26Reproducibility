! TSVC tsvc_2 s252:  a(i) = s(i) + s(i-1),  s(i) = b(i)*c(i),  s(0) := 0
!
! One-pass elementwise form (no loop-carried chain): each a(i) is
! independent, so the work splits cleanly across threads.
!
! NOTE on rounding: the compiler (gfortran, -ffp-contract=fast) is free to
! contract any product that feeds a single add into an FMA, which would
! change results by up to 1 ulp vs the numpy reference (which rounds each
! product, then rounds the add).  Here every product is used in TWO
! statements (one add and one store into scratch w), which makes FMA
! contraction impossible, so each product and each add is correctly
! rounded exactly like the reference.

module ktsvc
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine do_fused(a, b, c, w, tprev, ln)
    type(c_ptr), value :: a, b, c, w
    real(c_double), value :: tprev
    integer(c_int64_t), value :: ln
    real(c_double), dimension(:), pointer :: av, bv, cv, wv
    real(c_double) :: p1
    integer(c_int64_t) :: i
    call c_f_pointer(a, av, [ln])
    call c_f_pointer(b, bv, [ln])
    call c_f_pointer(c, cv, [ln])
    call c_f_pointer(w, wv, [ln + 1])
    p1 = bv(1) * cv(1)
    if (tprev == 0.0d0) then
      av(1) = p1
    else
      av(1) = p1 + tprev
    end if
    wv(2) = p1
    do i = 2, ln
       p1 = bv(i) * cv(i)
       av(i) = p1 + wv(i)
       wv(i+1) = p1
    end do
  end subroutine do_fused
end module ktsvc

subroutine tsvc_2_s252_fp64(a, b, c, n) bind(C, name="tsvc_2_s252_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  use ktsvc
  implicit none
  type(c_ptr), value :: a, b, c
  integer(c_int64_t), value :: n
  real(c_double), dimension(:), pointer :: av
  real(c_double), dimension(:), pointer :: bv
  real(c_double), dimension(:), pointer :: cv
  real(c_double), allocatable, target, save :: ws(:)
  integer(c_int64_t) :: lo, hi, ln, q, r, base, sz
  real(c_double) :: tprev
  integer :: nt, tid
  call c_f_pointer(a, av, [n])
  call c_f_pointer(b, bv, [n])
  call c_f_pointer(c, cv, [n])
  if (n < 1) return
  nt = omp_get_max_threads()
  sz = n + 3 * nt
  if (size(ws, 1) < n + 1) then
    if (allocated(ws)) deallocate(ws)
    allocate(ws(n + 3 * nt))
  end if
  if (n < 100000) then
    call do_fused(a, b, c, c_loc(ws(1)), 0.0d0, n)
    return
  end if
  !$omp parallel private(nt, tid, lo, hi, ln, q, r, base, tprev)
  nt = omp_get_num_threads()
  tid = omp_get_thread_num()
  q = n / nt
  r = n - q * nt
  if (tid < r) then
    lo = q * tid + tid + 1
    hi = lo + q
  else
    lo = (q + 1) * r + (tid - r) * q + 1
    hi = lo + q - 1
  end if
  ln = hi - lo + 1
  base = (q + 2) * tid + 1
  if (ln > 0) then
    if (lo > 1) then
      tprev = bv(lo-1) * cv(lo-1)
    else
      tprev = 0.0d0
    end if
    call do_fused(c_loc(av(lo)), c_loc(bv(lo)), c_loc(cv(lo)), &
                  c_loc(ws(base)), tprev, ln)
  end if
  !$omp end parallel
end subroutine tsvc_2_s252_fp64
