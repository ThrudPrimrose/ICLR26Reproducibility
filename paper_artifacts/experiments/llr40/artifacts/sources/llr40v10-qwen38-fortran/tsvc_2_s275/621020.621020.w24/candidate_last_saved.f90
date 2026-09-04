! TSVC tsvc_2 s275.
! C memory: A[j*L+i]. Reference: for each C column i: if A[0,i] > 0:
!   A[j,i] = A[j-1,i] + B[j,i]*C[j,i]  (scan along rows, stride L).
! Fortran view F(x,y) = C offset (x-1)*L+(y-1)  ->  F(j+1,i+1) = F(j,i+1)+...
! so: serial scan over x (rows) for fixed y (columns); y parallel.
subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(c, name="tsvc_2_s275_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in)    :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j
  real(c_double) :: carry

  if (len_2d < 2) return

  !$omp parallel do schedule(static)
  do i = 1, len_2d
     if (aa(1, i) > 0.0d0) then
        carry = aa(1, i)
        do j = 2, len_2d
           carry = carry + bb(j, i) * cc(j, i)
           aa(j, i) = carry
        end do
     end if
  end do
  !$omp end parallel do
end subroutine tsvc_2_s275_fp64
