subroutine tsvc_2_s115_fp64(a, aa, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D)
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i, j
  real(c_double), volatile :: prod
  !$omp parallel private(i, j, prod)
  do j = 1, LEN_2D - 1
    !$omp do simd schedule(static)
    do i = j + 1, LEN_2D
      prod = aa(i, j) * a(j)
        a(i) = a(i) - prod
    end do
    !$omp end do simd
  end do
  !$omp end parallel
end subroutine tsvc_2_s115_fp64
