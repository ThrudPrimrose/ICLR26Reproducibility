! TSVC tsvc_2 kernel s233 -- optimized Fortran (bit-identical to reference).
!
! Reference (C, 0-based): for i in 8..L: for j in 8..L: aa[j*L+i] = aa[(j-1)*L+i] + cc[j*L+i]
!                         for i in 8..L: for j in 8..L: bb[j*L+i] = bb[j*L+i-1] + cc[j*L+i]
!
! Fortran column-major: aa(i,j) sits at offset (i-1)+(j-1)*L, so C's aa[j][i] == Fortran aa(i,j):
!   (1) aa(i,j) = aa(i-1,j) + cc(i,j)   -- serial scan along i (contiguous), independent per column j (i,j = 9..L)
!       -> parallelize over j; each thread runs one contiguous dependency chain per column.
!   (2) bb(i,j) = bb(i,j-1) + cc(i,j)   -- serial scan along j (strided)
!       -> restructured: for each column j (outer), the whole column update
!            val(i) = val(i) + cc(i,j);  bb(i,j) = val(i)     (i = 8..L)
!          is a fully independent, vectorizable, contiguous loop; val(i) is the running
!          state val(i) == bb(i,j-1).  Each element performs the exact same floating-point
!          addition in the same order as the reference, so results are bit-identical.
!          val is persistent per thread over a contiguous block of columns.
subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(c, name="tsvc_2_s233_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j, l, tid, nt, n, c, extra, j0, j1
  real(c_double)     :: val(len_2d)

  l = len_2d

  ! (1) aa: scan along contiguous i, independent per column j.
  !$omp parallel do schedule(static)
  do j = 9, l
     do i = 9, l
        aa(i,j) = aa(i-1,j) + cc(i,j)
     end do
  end do

  ! (2) bb: column-wise restructured scan, one contiguous column block per thread.
  !$omp parallel shared(l, aa, bb, cc) private(tid, nt, n, c, extra, j0, j1, i, j, val)
     tid = omp_get_thread_num()
     nt = omp_get_num_threads()
     n = l - 8
     c = n / nt
     extra = n - c * nt
     j0 = 9 + tid * c + min(tid, extra)
     if (tid < extra) then
        j1 = j0 + c + 1
     else
        j1 = j0 + c
     end if
     if (j1 > j0) then
        do i = 9, l
           val(i) = bb(i, j0 - 1)
        end do
        do j = j0, j1 - 1
           do i = 9, l
              val(i) = val(i) + cc(i, j)
              bb(i, j) = val(i)
           end do
        end do
     end if
  !$omp end parallel

end subroutine tsvc_2_s233_fp64
