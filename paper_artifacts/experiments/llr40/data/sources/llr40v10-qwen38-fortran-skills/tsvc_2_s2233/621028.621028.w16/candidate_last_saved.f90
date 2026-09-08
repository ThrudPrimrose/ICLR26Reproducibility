! tsvc_2 s2233.  numpy reference:
!   for i in 8..n-1: for j in 8..n-1: aa[j,i] = aa[j-1,i] + cc[j,i]
!   for i in 8..n-1: for j in 8..n-1: bb[i,j] = bb[i-1,j] + cc[i,j]
! Fortran (column-major; numpy x[a,b] == Fortran x(b+1,a+1)):
!   part A:  aa(i,j) = aa(i,j-1) + cc(i,j),  i,j in 9..n   (chain on j, free on i)
!   part B:  bb(i,j) = bb(i,j-1) + cc(i,j),  i,j in 9..n   (chain on j, free on i)
! Both chains run along the SLOW (second) axis; the free axis is the FAST one.
! One parallel region: each thread owns a contiguous band of i (the free axis)
! and sweeps the chain j = 9..n.  All six streams are unit stride, and cc is
! read only ONCE because both updates consume the same cc element.
subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C, name="tsvc_2_s2233_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int64_t) :: n, i, j, nt, t, lo, hi

  n = len_2d
  if (n < 9) return
  nt = omp_get_max_threads()
  if (nt < 1) nt = 1

  !$omp parallel shared(aa, bb, cc, n, nt) private(t, lo, hi, i, j)
    t = omp_get_thread_num()
    lo = (n - 8) * t / nt + 9
    hi = (n - 8) * (t + 1) / nt + 8

    do j = 9, n
      do i = lo, hi
        aa(i, j) = aa(i, j - 1) + cc(i, j)
        bb(i, j) = bb(i, j - 1) + cc(i, j)
      end do
    end do
  !$omp end parallel
end subroutine
