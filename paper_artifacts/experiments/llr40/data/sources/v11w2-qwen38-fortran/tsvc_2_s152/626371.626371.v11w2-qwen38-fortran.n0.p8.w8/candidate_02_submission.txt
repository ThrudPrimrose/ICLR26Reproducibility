! TSVC tsvc_2 kernel s152, fp64:  b = d*e ; a += b*c
!
! Memory-bound streaming kernel. Work is split in 16-element blocks so each
! vector lane keeps a pair of independent 8-wide AVX-512 streams in flight
! (higher load parallelism than the plain stride-1 loop). Array-section
! statements of fixed length 8 are what the compiler vectorizes safely --
! an explicit-stride loop (do i=1,n,16) with +8 offsets miscompiles in this
! gfortran, so it is avoided.
!
! ABI notes (gfortran 16 BIND(C)): the scalar length dummy must carry VALUE
! or it arrives as a reference; arrays are plain assumed-size pointers.
subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s152_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double) :: a(*), b(*), c(*), d(*), e(*)
  integer(c_int64_t), value :: len_1d
  integer(c_int64_t) :: i, n16
  intent(inout) :: a, b
  intent(in)    :: c, d, e, len_1d

  n16 = (len_1d/16)*16
!$omp parallel do schedule(static)
  do i = 1, n16, 16
    b(i:i+7)   = d(i:i+7)   * e(i:i+7)
    b(i+8:i+15) = d(i+8:i+15) * e(i+8:i+15)
    a(i:i+7)   = a(i:i+7)   + b(i:i+7)   * c(i:i+7)
    a(i+8:i+15) = a(i+8:i+15) + b(i+8:i+15) * c(i+8:i+15)
  end do
!$omp end parallel do
!$omp parallel do schedule(static)
  do i = n16 + 1, len_1d
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
!$omp end parallel do
end subroutine tsvc_2_s152_fp64
