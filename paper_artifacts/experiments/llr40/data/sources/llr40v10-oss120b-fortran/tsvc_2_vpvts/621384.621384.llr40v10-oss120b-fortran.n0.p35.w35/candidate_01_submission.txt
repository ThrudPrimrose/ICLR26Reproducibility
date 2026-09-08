module tsvc_2_vpvts_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C, name="tsvc_2_vpvts_fp64")
    ! Arguments
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    integer(c_int64_t), value    :: LEN_1D
    integer(c_int64_t), value    :: S
    integer(c_int64_t) :: i
    ! Parallel loop with SIMD
    !$omp parallel do default(none) shared(a, b, S, LEN_1D) schedule(static) private(i)
    do i = 1, LEN_1D
      a(i) = a(i) + b(i) * dble(S)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_vpvts_fp64
end module tsvc_2_vpvts_mod
