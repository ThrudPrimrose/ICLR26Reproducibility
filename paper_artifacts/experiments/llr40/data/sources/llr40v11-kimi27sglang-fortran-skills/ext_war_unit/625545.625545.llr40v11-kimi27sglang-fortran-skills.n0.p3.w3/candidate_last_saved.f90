subroutine ext_war_unit_fp64(a, b, n, workspace, workspace_size) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n)
  integer(c_int8_t), target, intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  real(c_double), pointer :: t(:)

  if (workspace_size >= 8 * (n - 1)) then
    call c_f_pointer(c_loc(workspace(1)), t, [n - 1])
    !$omp parallel private(i)
    !$omp do simd schedule(static)
    do i = 2, n
      t(i - 1) = a(i)
    end do
    !$omp end do simd
    !$omp do simd schedule(static)
    do i = 1, n - 1
      a(i) = t(i) + b(i)
    end do
    !$omp end do simd
    !$omp end parallel
  else
    !$omp simd
    do i = 1, n - 1
      a(i) = a(i + 1) + b(i)
    end do
    !$omp end simd
  end if
end subroutine ext_war_unit_fp64
