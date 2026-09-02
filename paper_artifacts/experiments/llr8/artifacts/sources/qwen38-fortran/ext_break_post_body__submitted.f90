! TSVC s482: a[i] = a[i] + b[i]*c[i]; break at first i with c[i] > b[i] (write kept).
! Strategy: parallel first-hit search over the mask c > b (min reduction), then a
! single parallel, vectorizable FMA pass over the prefix [1..L].

subroutine ext_break_post_body_fp64(a, b, c, n) bind(C, name='ext_break_post_body_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  integer(kind=8), intent(in), value :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in)    :: b(n)
  real(c_double), intent(in)    :: c(n)

  integer(kind=8) :: i, L, lb

  if (n <= 0) return

  ! Small sizes: no parallelism -- plain fused loop, exactly the sequential semantics.
  if (n < 8192) then
    do i = 1, n
      a(i) = a(i) + b(i) * c(i)
      if (c(i) > b(i)) return
    end do
    return
  end if

  ! Phase A: first index where c(i) > b(i); lb stays n+1 if there is none.
  ! Per-thread: min hit in its chunk; chunk min == first hit in chunk; global min
  ! of chunk firsts == global first hit.
  lb = n + 1
  !$omp parallel do schedule(static) reduction(min:lb) default(none) shared(b, c, n)
  do i = 1, n
    if (c(i) > b(i) .and. i < lb) lb = i
  end do
  !$omp end parallel do

  L = min(lb, n)

  ! Phase B: fused multiply-add over the inclusive prefix.
  if (L > 0) then
    !$omp parallel do schedule(static) default(none) shared(a, b, c, L)
    do i = 1, L
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do
  end if
end subroutine ext_break_post_body_fp64
