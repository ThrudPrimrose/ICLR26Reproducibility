subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN, workspace, workspace_size) bind(C, name="tsvc_2_s1232_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, i_start

  !$omp parallel do default(none) shared(aa, bb, cc, LEN_2D, VLEN) private(i, i_start) schedule(static,1)
  do j = 1, LEN_2D
    i_start = (j - 1) * VLEN + 1
    if (i_start > LEN_2D) cycle
    !$omp simd
    do i = i_start, LEN_2D
      aa(j, i) = bb(j, i) + cc(j, i)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s1232_fp64
