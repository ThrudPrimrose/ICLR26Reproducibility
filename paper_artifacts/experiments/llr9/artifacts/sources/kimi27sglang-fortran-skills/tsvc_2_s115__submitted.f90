subroutine tsvc_2_s115_fp64(a, aa, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(in) :: aa(len_2d, len_2d)
  integer(c_int64_t) :: i, j
  real(c_double) :: aj

  !$omp parallel private(i, j, aj)
  do j = 1, len_2d
    aj = a(j)
    !$omp do simd schedule(static)
    do i = j + 1, len_2d
      a(i) = a(i) - aa(i, j) * aj
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s115_fp64
