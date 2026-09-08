! wf_diff_skew -- difference-diagonal wavefront
!   a[i,j] = a[i,j] + a[i-1,j] + a[i-1,j+1]
! Row-major LEN_2D x LEN_2D double array (C ABI).
! Rows are strictly sequential (row i reads updated row i-1); within a row
! all j are independent.  The inner loop is unit-stride in all three arrays,
! so gfortran emits a clean AVX-512 8-wide body whose row load/store share a
! single base register (matching a hand-written C kernel).
subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  real(c_double) :: a(*)
  integer(c_int64_t), value :: LEN_2D
  integer(c_int64_t) :: i, j, n

  n = LEN_2D
  do i = 1, n-1
    do j = 1, n-1
      a(i*n+j) = a(i*n+j) + a((i-1)*n+j) + a((i-1)*n+j+1)
    end do
  end do
end subroutine wf_diff_skew_fp64
