module ext_break_post_body_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine ext_break_post_body_fp64(a, b, c, LEN_1D, workspace, workspace_size) bind(C, name='ext_break_post_body_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: break_idx
    break_idx = LEN_1D + 1
    !$omp parallel do default(none) shared(b, c, LEN_1D) private(i) reduction(min:break_idx)
    do i = 1, LEN_1D
        if (c(i) > b(i)) then
            break_idx = i
        end if
    end do
    !$omp end parallel do
    if (break_idx > LEN_1D) then
        break_idx = LEN_1D
    end if
    a(1:break_idx) = a(1:break_idx) + b(1:break_idx) * c(1:break_idx)
  end subroutine ext_break_post_body_fp64
end module ext_break_post_body_mod
