module tsvc_2_s316_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C, name="tsvc_2_s316_fp64")
    ! Arguments
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    integer(c_int64_t), value :: LEN_1D
    ! Local variables
    real(c_double) :: x
    integer(c_int64_t) :: i

    if (LEN_1D <= 0) then
      result(1) = 0.0_c_double
      return
    end if

    x = a(1)
    !$omp parallel do reduction(min:x) schedule(static)
    do i = 2, LEN_1D
      x = min(x, a(i))
    end do
    !$omp end parallel do
    result(1) = x
  end subroutine tsvc_2_s316_fp64
end module tsvc_2_s316_mod
