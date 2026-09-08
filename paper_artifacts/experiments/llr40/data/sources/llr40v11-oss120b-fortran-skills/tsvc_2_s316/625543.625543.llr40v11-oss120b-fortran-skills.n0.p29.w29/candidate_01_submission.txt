module tsvc_2_s316_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C, name="tsvc_2_s316_fp64")
    implicit none
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(1)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: x
    integer(c_int64_t) :: i

    x = huge(0.0_c_double)
    !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(min:x)
    do i = 1, LEN_1D
      if (a(i) < x) x = a(i)
    end do
    !$omp end parallel do
    result(1) = x
  end subroutine tsvc_2_s316_fp64
end module tsvc_2_s316_mod
