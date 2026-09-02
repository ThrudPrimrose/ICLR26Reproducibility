subroutine ext_war_unit_fp64(a, b, LEN_1D, workspace, workspace_size) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  ! Arguments: double* a, double* b, int64_t LEN_1D, uint8_t* workspace, int64_t workspace_size
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_size
  integer(c_int64_t) :: i

  if (LEN_1D > 1) then
    !$omp simd
    do i = 1, LEN_1D-1
      a(i) = a(i+1) + b(i)
    end do
    ! No explicit end directive needed for !$omp simd
  end if
end subroutine ext_war_unit_fp64
