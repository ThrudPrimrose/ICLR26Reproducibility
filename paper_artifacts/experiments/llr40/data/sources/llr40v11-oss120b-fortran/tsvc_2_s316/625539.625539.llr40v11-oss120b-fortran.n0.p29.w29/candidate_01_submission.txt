module tsvc_2_s316_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name="tsvc_2_s316_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    integer(c_int64_t), value :: len_1d
    real(c_double) :: x
    integer(c_int64_t) :: i
    x = huge(0.0_c_double)
    !$omp parallel do reduction(min:x) schedule(static)
    do i = 1, len_1d
      if (a(i) < x) then
        x = a(i)
      end if
    end do
    !$omp end parallel do
    result(1) = x
  end subroutine tsvc_2_s316_fp64
end module tsvc_2_s316_mod
