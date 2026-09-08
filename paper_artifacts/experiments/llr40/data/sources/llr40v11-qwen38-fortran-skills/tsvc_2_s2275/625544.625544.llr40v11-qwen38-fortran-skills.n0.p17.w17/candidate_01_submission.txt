subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  real(c_double), intent(in) :: d(len_2d)
  integer(c_int64_t) :: j

  ! aa(j,i) += bb(j,i)*cc(j,i) for all (i,j)  -- elementwise, no dependences.
  ! Fortran first subscript is the unit-stride (column) axis: the C flat index
  ! idx = j*LEN_2D + i is the Fortran element aa(i, j).
  ! The vector update a(i) = b(i) + c(i)*d(i) is fused into the column loop
  ! (disjoint arrays, no dependence).
  !$omp parallel do
  do j = 1, len_2d
    aa(:, j) = aa(:, j) + bb(:, j) * cc(:, j)
    a(j) = b(j) + c(j) * d(j)
  end do
end subroutine tsvc_2_s2275_fp64
