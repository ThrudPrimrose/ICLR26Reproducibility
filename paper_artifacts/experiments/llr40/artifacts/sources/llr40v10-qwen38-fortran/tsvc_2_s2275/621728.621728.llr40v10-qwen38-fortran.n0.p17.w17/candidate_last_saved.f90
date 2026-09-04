! TSVC tsvc_2 s2275 (fp64):
!   C flat idx = j*len + i :  aa[idx] += bb[idx]*cc[idx]  for all i,j
!                             a[i]    = b[i] + c[i]*d[i]  for all i
! Fortran column-major: aa(i,j) lives at (j-1)*len + (i-1), so looping i
! inner is unit stride. The update is elementwise, loop order is irrelevant.
subroutine tsvc_2_s2275_fp64(a, aa, b, bb, cv, cc, d, len_2d) bind(C, name="tsvc_2_s2275_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), dimension(len_2d), intent(out) :: a
  real(c_double), dimension(len_2d, len_2d), intent(inout) :: aa
  real(c_double), dimension(len_2d), intent(in) :: b
  real(c_double), dimension(len_2d, len_2d), intent(in) :: bb
  real(c_double), dimension(len_2d, len_2d), intent(in) :: cc
  real(c_double), dimension(len_2d), intent(in) :: d
  real(c_double), dimension(len_2d), intent(in) :: cv
  integer(c_int64_t) :: i, j

  !$omp parallel
  !$omp do
  do j = 1, len_2d
     do i = 1, len_2d
        aa(i, j) = aa(i, j) + bb(i, j) * cc(i, j)
     end do
     a(j) = b(j) + cv(j) * d(j)
  end do
  !$omp end parallel
end subroutine tsvc_2_s2275_fp64
