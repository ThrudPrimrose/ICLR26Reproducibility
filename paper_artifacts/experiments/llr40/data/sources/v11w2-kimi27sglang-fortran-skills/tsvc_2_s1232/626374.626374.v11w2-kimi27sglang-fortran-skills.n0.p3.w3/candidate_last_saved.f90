subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN, workspace_size
  type(c_ptr), value, intent(in) :: workspace
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, jmax

  !$omp parallel do private(j, jmax) proc_bind(spread) schedule(guided)
  do i = 1, LEN_2D
    jmax = min(LEN_2D, (i - 1) / VLEN + 1)
    !$omp simd
    do j = 1, jmax
      aa(j, i) = bb(j, i) + cc(j, i)
    end do
    !$omp end simd
  end do
  !$omp end parallel do
end subroutine tsvc_2_s1232_fp64
