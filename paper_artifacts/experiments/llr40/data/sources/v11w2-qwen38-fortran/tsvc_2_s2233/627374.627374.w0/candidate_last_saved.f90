! tsvc_2 s2233: two column-wise prefix scans over LEN_2D x LEN_2D fp64 arrays.
! C reference: for i=8..L-1: for j=8..L-1: aa[j*L+i] = aa[(j-1)*L+i] + cc[j*L+i]
!                        for j=8..L-1: bb[i*L+j] = bb[(i-1)*L+j] + cc[i*L+j]
! Both are scans along C-row (flat 1st index) for each column, vectorized across
! 8 contiguous columns. Each thread handles a 64-column unit (8 blocks) so its
! row stores are mostly full cache-line writes (row stride is not a multiple of
! 64B; adjacent small windows would co-own cache lines across threads).

subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(c, name="tsvc_2_s2233_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value :: len_2d
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in)    :: cc(*)

  integer(c_int64_t) :: nunit, rem, nblk, nsc, c0, base, u, r, tnb
  integer :: j, k, tn
  real(c_double), allocatable :: tbuf(:)

  if (len_2d <= 8) return
  nunit = (len_2d - 8) / 64
  rem = len_2d - 8 - 64 * nunit
  nblk = rem / 8
  nsc = rem - 8 * nblk
  allocate(tbuf(2 * 64 * 1024))

  !$omp parallel

     ! aa: 64-column units
     !$omp do schedule(static)
     do u = 1, nunit
        tnb = 128 * omp_get_thread_num()
        base = 64 * (u - 1) + 8
        do j = 0, 7
           do k = 1, 8
              tbuf(tnb + 8 * j + k) = aa(7 * len_2d + base + 8 * j + k)
           end do
        end do
        do r = 8, len_2d - 1
           do j = 0, 7
              !$omp simd
              do k = 1, 8
                 tbuf(tnb + 8 * j + k) = tbuf(tnb + 8 * j + k) + cc(r * len_2d + base + 8 * j + k)
                 aa(r * len_2d + base + 8 * j + k) = tbuf(tnb + 8 * j + k)
              end do
           end do
        end do
     end do
     !$omp end do

     ! aa: remainder 8-column blocks
     !$omp do schedule(static)
     do u = 1, nblk
        tnb = 128 * omp_get_thread_num()
        c0 = 64 * nunit + 8 * u
        do k = 1, 8
           tbuf(tnb + k) = aa(7 * len_2d + c0 + k)
        end do
        do r = 8, len_2d - 1
           !$omp simd
           do k = 1, 8
              tbuf(tnb + k) = tbuf(tnb + k) + cc(r * len_2d + c0 + k)
              aa(r * len_2d + c0 + k) = tbuf(tnb + k)
           end do
        end do
     end do
     !$omp end do

     ! bb: 64-column units
     !$omp do schedule(static)
     do u = 1, nunit
        tnb = 128 * omp_get_thread_num()
        base = 64 * (u - 1) + 8
        do j = 0, 7
           do k = 1, 8
              tbuf(tnb + 64 + 8 * j + k) = bb(7 * len_2d + base + 8 * j + k)
           end do
        end do
        do r = 8, len_2d - 1
           do j = 0, 7
              !$omp simd
              do k = 1, 8
                 tbuf(tnb + 64 + 8 * j + k) = tbuf(tnb + 64 + 8 * j + k) + cc(r * len_2d + base + 8 * j + k)
                 bb(r * len_2d + base + 8 * j + k) = tbuf(tnb + 64 + 8 * j + k)
              end do
           end do
        end do
     end do
     !$omp end do

     ! bb: remainder 8-column blocks
     !$omp do schedule(static)
     do u = 1, nblk
        tnb = 128 * omp_get_thread_num()
        c0 = 64 * nunit + 8 * u
        do k = 1, 8
           tbuf(tnb + 64 + k) = bb(7 * len_2d + c0 + k)
        end do
        do r = 8, len_2d - 1
           !$omp simd
           do k = 1, 8
              tbuf(tnb + 64 + k) = tbuf(tnb + 64 + k) + cc(r * len_2d + c0 + k)
              bb(r * len_2d + c0 + k) = tbuf(tnb + 64 + k)
           end do
        end do
     end do
     !$omp end do

     ! remainder scalar columns (aa and bb)
     !$omp do schedule(static)
     do u = 1, nsc
        c0 = 64 * nunit + 8 * nblk + 8 + u - 1
        do r = 8, len_2d - 1
           aa(r * len_2d + c0 + 1) = aa((r - 1) * len_2d + c0 + 1) + cc(r * len_2d + c0 + 1)
        end do
     end do
     !$omp end do

     !$omp do schedule(static)
     do u = 1, nsc
        c0 = 64 * nunit + 8 * nblk + 8 + u - 1
        do r = 8, len_2d - 1
           bb(r * len_2d + c0 + 1) = bb((r - 1) * len_2d + c0 + 1) + cc(r * len_2d + c0 + 1)
        end do
     end do
     !$omp end do
  !$omp end parallel
  deallocate(tbuf)
end subroutine tsvc_2_s2233_fp64
