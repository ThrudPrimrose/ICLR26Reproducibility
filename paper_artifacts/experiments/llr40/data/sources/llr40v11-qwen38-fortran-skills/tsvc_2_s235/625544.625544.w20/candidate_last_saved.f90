! TSVC tsvc_2 s235:
!   for i in 0..N-1:
!     a[i] += b[i]*c[i]
!     for j in 1..N-1:
!       aa[j,i] = aa[j-1,i] + bb[j,i]*a[i]
!
! Fortran mapping (column-major): aa(j,i)_C = aa(i,j)_F, bb(j,i)_C = bb(i,j)_F.
! Dependence: (0,+1) along j only -> outer i loop is fully parallel.

subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)

  integer(c_int64_t) :: i, j

  !$omp parallel do schedule(static)
  do i = 1, len_2d
    a(i) = a(i) + b(i) * c(i)
    do j = 2, len_2d
      aa(i, j) = aa(i, j - 1) + bb(i, j) * a(i)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s235_fp64
