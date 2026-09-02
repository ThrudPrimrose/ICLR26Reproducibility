subroutine ext_war_unit_fp64(a, b, LEN_1D, workspace, workspace_bytes) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value :: LEN_1D
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i

  if (LEN_1D > 1) then
    !$omp simd
    do i = 1, LEN_1D-1
      a(i) = a(i+1) + b(i)
    end do
  end if
end subroutine ext_war_unit_fp64
