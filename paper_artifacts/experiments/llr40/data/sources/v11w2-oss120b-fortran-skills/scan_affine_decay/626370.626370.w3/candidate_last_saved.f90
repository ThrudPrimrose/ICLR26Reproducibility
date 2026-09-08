module scan_affine_decay_mod
  use iso_c_binding, only: c_int64_t, c_double, c_ptr
  implicit none
contains
  subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_size) bind(C, name="scan_affine_decay_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: y(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i

    if (LEN_1D <= 1_c_int64_t) return

    !$omp parallel do if(.false.) default(none) shared(y,c,x,LEN_1D) private(i)
    do i = 2, LEN_1D
      y(i) = c(i) * y(i-1) + x(i)
    end do
!$omp end parallel do

  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
