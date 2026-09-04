! TSVC tsvc_2 s1232 (fp64), v2 C-ABI (pointer args by value, like the C reference).
!
! Reference (1-D buffer positions i*N + j):
!   for (j = 0; j < N; ++j)
!     for (i = j*VLEN; i < N; ++i)
!       aa[i*N + j] = bb[i*N + j] + cc[i*N + j]
!
! j is the unit-stride dimension, i the strided one.  Each element (i,j)
! is written by at most one iteration and bb/cc are never modified, so the
! loop carries NO dependence.  The updated set is
!   {(i,j) : 0 <= j < N, j*VLEN <= i < N}
! and for fixed i the updated j-range is the contiguous prefix
!   j = 0 .. min(N, i/VLEN) - 1.
! We iterate i in parallel; the inner loop is unit-stride -> vectorizes.
subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(c, name="tsvc_2_s1232_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: aa
  real(c_double), dimension(*), intent(in)    :: bb, cc
  integer(c_int64_t), value, intent(in)       :: len_2d, vlen
  integer(c_int64_t) :: n, v, i, jm, base
  integer(c_int64_t) :: j

  n = len_2d
  if (n <= 0) return
  v = vlen

  if (v <= 0) then
     ! VLEN = 0: every column is fully updated (i starts at 0).
     !$omp parallel do schedule(static, 1)
     do i = 1, n
        base = (i - 1) * n
        do j = 1, n
           aa(base + j) = bb(base + j) + cc(base + j)
        end do
     end do
  else
     !$omp parallel do schedule(static, 1)
     do i = 1, n
        base = (i - 1) * n
        jm = min(n, (i - 1) / v + 1)
        do j = 1, jm
           aa(base + j) = bb(base + j) + cc(base + j)
        end do
     end do
  end if
end subroutine tsvc_2_s1232_fp64
