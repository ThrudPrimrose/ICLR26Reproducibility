subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d), bb(len_2d, len_2d), c(len_2d), cc(len_2d, len_2d), d(len_2d)
  real(c_double) :: workspace(workspace_size)
  real(c_double), pointer :: pa1(:), pb1(:), pc1(:)
  integer :: i
  integer(c_int64_t) :: total, k

  pa1 => aa
  pb1 => bb
  pc1 => cc
  total = len_2d * len_2d

  !$omp parallel do schedule(static)
  do k = 1, total
    pa1(k) = pa1(k) + pb1(k) * pc1(k)
  end do

  !$omp parallel do simd
  do i = 1, len_2d
    a(i) = b(i) + c(i) * d(i)
  end do
end subroutine tsvc_2_s2275_fp64
