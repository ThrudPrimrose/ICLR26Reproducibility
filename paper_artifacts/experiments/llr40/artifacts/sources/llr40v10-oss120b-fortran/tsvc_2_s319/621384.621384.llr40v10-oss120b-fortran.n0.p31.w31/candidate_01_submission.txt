module tsvc_2_s319_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s319_fp64")
    ! Arguments correspond to C signature:
    ! void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
    !   const double *restrict c, const double *restrict d,
    !   const double *restrict e, const int64_t LEN_1D)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    integer(c_int64_t), value :: LEN_1D

    integer(c_int64_t) :: i
    real(c_double) :: sum

    sum = 0.0_c_double
    ! Parallelize over the 1D domain with reduction on sum.
    !$omp parallel do default(none) shared(a,b,c,d,e,LEN_1D) private(i) reduction(+:sum) schedule(static)
    do i = 0, LEN_1D - 1
      a(i+1) = c(i+1) + d(i+1)
      sum = sum + a(i+1)
      b(i+1) = c(i+1) + e(i+1)
      sum = sum + b(i+1)
    end do
    !$omp end parallel do
    b(1) = sum
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
