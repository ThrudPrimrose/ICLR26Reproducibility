module tsvc_2_vpvts_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C, name="tsvc_2_vpvts_fp64")
    ! Arguments: a - inout double array, b - in double array, LEN_1D - length, S - scalar multiplier
    implicit none
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t), value :: S
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t) :: i
    ! Convert S to double once to avoid repeated conversion (optional)
    real(c_double) :: s_real
    s_real = real(S, kind=c_double)
    !$omp parallel do default(none) shared(a,b, LEN_1D, s_real) private(i)
    do i = 1, LEN_1D
      a(i) = a(i) + b(i) * s_real
    end do
    !$omp end parallel do
  end subroutine tsvc_2_vpvts_fp64
end module tsvc_2_vpvts_mod
