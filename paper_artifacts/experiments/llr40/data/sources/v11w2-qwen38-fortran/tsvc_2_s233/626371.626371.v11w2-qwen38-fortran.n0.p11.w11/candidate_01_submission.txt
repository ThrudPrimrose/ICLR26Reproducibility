! TSVC tsvc_2 s233, fp64, C-ABI symbol tsvc_2_s233_fp64.
!
! Reference semantics (C indexing, flat row-major):
!   for i = 8..n-1:
!     for j = 8..n-1: aa[j*n+i] = aa[(j-1)*n+i] + cc[j*n+i]  (scan down column i)
!     for j = 8..n-1: bb[j*n+i] = bb[j*n+(i-1)] + cc[j*n+i]  (scan along row j)
! The two parts touch disjoint outputs and only read cc, so they are fully
! independent and are run as two parallel phases. Fortran subscripts are
! 1-based, so every C position t is addressed as t+1.
!
! Part 1 (aa): each unit owns 8 consecutive columns = exactly one 64-byte cache
! line per row, so no two threads ever touch the same line (no false sharing).
! The 8 columns are independent chains, interleaved in one k-loop so the
! out-of-order engine pipelines them; the k-loop vectorizes.
! Part 2 (bb): each unit owns 8 consecutive rows, scanning i left to right;
! the 8 rows are independent chains interleaved in the k-loop.
subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C, name="tsvc_2_s233_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: aa, bb
  real(c_double), dimension(*), intent(in)    :: cc
  integer(c_int64_t), value, intent(in)       :: len_2d
  integer(c_int64_t) :: n, i0, j0, j, i, k, nu, rem

  n = len_2d

  ! ---- part 1: aa[j*n+i] = aa[(j-1)*n+i] + cc[j*n+i] ----
  ! units of 8 columns starting at column 8 + 8*nu
  nu = (n - 8) / 8
  if (nu > 0) then
    !$omp parallel do default(none) shared(aa,cc,n,nu) private(i0,j,k) schedule(static)
    do i0 = 8, 8*nu + 7, 8
      do j = 8, n - 1
        do k = 0, 7
          aa(j*n + i0 + k + 1) = aa((j-1)*n + i0 + k + 1) + cc(j*n + i0 + k + 1)
        end do
      end do
    end do
    !$omp end parallel do
  end if
  rem = n - 8 - 8*nu
  if (rem > 0) then
    !$omp parallel do default(none) shared(aa,cc,n,nu,rem) private(i0,j) schedule(static)
    do i0 = 8 + 8*nu, n - 1
      do j = 8, n - 1
        aa(j*n + i0 + 1) = aa((j-1)*n + i0 + 1) + cc(j*n + i0 + 1)
      end do
    end do
    !$omp end parallel do
  end if

  ! ---- part 2: bb[j*n+i] = bb[j*n+(i-1)] + cc[j*n+i] ----
  ! units of 8 rows starting at row 8 + 8*nu
  if (nu > 0) then
    !$omp parallel do default(none) shared(bb,cc,n,nu) private(j0,i,k) schedule(static)
    do j0 = 8, 8*nu + 7, 8
      do i = 8, n - 1
        do k = 0, 7
          bb((j0+k)*n + i + 1) = bb((j0+k)*n + i) + cc((j0+k)*n + i + 1)
        end do
      end do
    end do
    !$omp end parallel do
  end if
  if (rem > 0) then
    !$omp parallel do default(none) shared(bb,cc,n,nu,rem) private(j0,i) schedule(static)
    do j0 = 8 + 8*nu, n - 1
      do i = 8, n - 1
        bb(j0*n + i + 1) = bb(j0*n + i) + cc(j0*n + i + 1)
      end do
    end do
    !$omp end parallel do
  end if

end subroutine tsvc_2_s233_fp64
