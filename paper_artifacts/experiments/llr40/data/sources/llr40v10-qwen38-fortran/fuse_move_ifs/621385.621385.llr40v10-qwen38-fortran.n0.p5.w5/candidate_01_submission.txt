! Optimized TSVC tsvc_2_5 fuse_move_ifs.
!
! Reference semantics (0-based C indexing):
!   for i in 0..N-1: if cond(i) > 0 then a[i,j] = 2*src[i,j] for all j
!   if K > 0 then b[i,j] = src[i,j] + 1 for all i,j
!
! Both nests write disjoint arrays, so they are fused into a single
! row-parallel pass that reads src once per row. The guards are per-row
! (scalar), so the inner loops stay straight-line and vectorize.
!
! C-interop note: bind(C) assumed-size dummies are 1-based, so C offset
! (r*n + c) is reached as x(r*n + c + 1). Row loop i = 1..n therefore
! covers C row i-1, and cond(i) reads exactly C's cond(i-1).
subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name='fuse_move_ifs_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(in)    :: cond(*)
  real(c_double), intent(in)    :: src(*)
  integer(c_int64_t), value     :: K
  integer(c_int64_t), value     :: LEN_2D

  integer(c_int64_t) :: i, g, n, base
  logical            :: kpos

  n = LEN_2D
  kpos = K > 0

  !$omp parallel do schedule(static) default(none) shared(a, b, cond, src, n, kpos) private(i, g, base)
  do i = 1, n
    base = (i - 1) * n
    if (cond(i) > 0.0d0) then
      if (kpos) then
        do g = 1, n
          a(base + g) = 2.0d0 * src(base + g)
          b(base + g) = src(base + g) + 1.0d0
        end do
      else
        do g = 1, n
          a(base + g) = 2.0d0 * src(base + g)
        end do
      end if
    else if (kpos) then
      do g = 1, n
        b(base + g) = src(base + g) + 1.0d0
      end do
    end if
  end do
  !$omp end parallel do
end subroutine fuse_move_ifs_fp64
