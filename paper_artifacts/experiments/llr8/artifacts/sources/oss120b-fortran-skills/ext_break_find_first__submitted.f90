module ext_break_find_first_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_find_first_fp64(a, b, c, d, LEN_1D, workspace, workspace_size) bind(C, name="ext_break_find_first_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D), d(LEN_1D)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: i_break
    i_break = LEN_1D
    do i = 1, LEN_1D
      if (d(i) < 0.0d0) then
        i_break = i - 1
        exit
      end if
    end do
    if (i_break >= 1) then
!$omp parallel do default(none) shared(a,b,c,i_break) private(i) schedule(static)
      do i = 1, i_break
        a(i) = a(i) + b(i) * c(i)
      end do
!$omp end parallel do
    end if
  end subroutine ext_break_find_first_fp64
end module ext_break_find_first_mod