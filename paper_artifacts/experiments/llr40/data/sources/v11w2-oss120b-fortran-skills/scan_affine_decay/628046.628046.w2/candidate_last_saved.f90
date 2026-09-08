module scan_affine_decay_mod
  use iso_c_binding
  implicit none
contains
  subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_size) bind(C)
    type(c_ptr), value, intent(in) :: y
    integer(c_int64_t), intent(in) :: LEN_1D
    real(c_double), intent(in) :: c(LEN_1D), x(LEN_1D)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), pointer :: y_f(:)
    integer(c_int64_t) :: i

    call c_f_pointer(y, y_f, [LEN_1D])

    if (LEN_1D == 0_c_int64_t) return
    y_f(1) = x(1)
    if (LEN_1D == 1_c_int64_t) return

    do i = 2_c_int64_t, LEN_1D
        y_f(i) = c(i) * y_f(i-1) + x(i)
    end do

  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
