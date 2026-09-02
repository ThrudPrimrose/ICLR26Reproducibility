subroutine tsvc_2_s232_fp64(aa, bb, len_2d) bind(C, name="tsvc_2_s232_fp64")
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int64_t) :: i, j

  !$omp parallel do private(i) schedule(static)
  do j = 2, len_2d
    do i = 2, j
      aa(i, j) = aa(i-1, j) * aa(i-1, j) + bb(i, j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s232_fp64
