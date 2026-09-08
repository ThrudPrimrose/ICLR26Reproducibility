! tsvc_2 s233:
!   for i = 8..n-1: for j = 8..n-1: aa[j,i] = aa[j-1,i] + cc[j,i]   (C order)
!   for i = 8..n-1: for j = 8..n-1: bb[j,i] = bb[j,i-1] + cc[j,i]   (C order)
!
! Memory layout (C order): element (j,i) at offset j*n + i.
!  - Part 1 (aa): recurrence along j, independent along i.
!    -> parallel over i in blocks of 8 (contiguous in memory), serial j-scan with
!       8 independent SIMD lanes (carries in a zmm register).
!  - Part 2 (bb): recurrence along i, independent along j.
!    -> parallel over j (rows); each row is a contiguous serial scan.
! Same operation order per element as the reference => bit-identical results.

subroutine tsvc_2_s233_fp64(aa, bb, cc, len2d) bind(C, name='tsvc_2_s233_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), intent(in)             :: aa, bb, cc
  integer(c_int64_t), intent(in)      :: len2d

  real(c_double), pointer :: a2d(:,:) => null()
  real(c_double), pointer :: b2d(:,:) => null()
  real(c_double), pointer :: c2d(:,:) => null()
  integer :: n, i, j, blk, k
  real(c_double) :: acc(0:7), r
  integer :: nblk, rem, base

  n = int(len2d)
  if (n <= 0) return

  call c_f_pointer(aa, a2d, shape=[n, n])
  call c_f_pointer(bb, b2d, shape=[n, n])
  call c_f_pointer(cc, c2d, shape=[n, n])

  if (n <= 8) return

  nblk = (n - 8) / 8
  rem  = mod(n - 8, 8)

  ! ---------------- Part 1: aa ----------------
  ! a2d(i,j) = a2d(i,j-1) + c2d(i,j);  i = first (contiguous) index.
  !$omp parallel do schedule(static)
  do blk = 0, nblk - 1
    base = 8 + blk * 8
    do k = 0, 7
      acc(k) = a2d(base + k, 7)
    end do
    do j = 8, n - 1
      do k = 0, 7
        acc(k) = acc(k) + c2d(base + k, j)
        a2d(base + k, j) = acc(k)
      end do
    end do
  end do
  !$omp end parallel do

  if (rem > 0) then
    base = 8 + nblk * 8
    do k = 0, rem - 1
      r = a2d(base + k, 7)
      do j = 8, n - 1
        r = r + c2d(base + k, j)
        a2d(base + k, j) = r
      end do
    end do
  end if

  ! ---------------- Part 2: bb ----------------
  ! b2d(i,j) = b2d(i-1,j) + c2d(i,j); rows j are independent.
  !$omp parallel do schedule(static)
  do j = 8, n - 1
    r = b2d(7, j)
    do i = 8, n - 1
      r = r + c2d(i, j)
      b2d(i, j) = r
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s233_fp64
