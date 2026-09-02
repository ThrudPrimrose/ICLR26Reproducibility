subroutine tsvc_2_s232_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, workspace_size
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: j, i

  if (LEN_2D < 128) then
    do j = 2, LEN_2D
      do i = 2, j
        aa(i, j) = aa(i - 1, j) * aa(i - 1, j) + bb(i, j)
      end do
    end do
    return
  end if

  !$omp parallel do schedule(guided) private(i) proc_bind(close)
  do j = 2, LEN_2D
    do i = 2, j
      aa(i, j) = aa(i - 1, j) * aa(i - 1, j) + bb(i, j)
 end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s232_fp64
