subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d, workspace, workspace_size) &
  bind(C, name="tsvc_2_s152_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_1d
  integer(c_int64_t), value :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  real(c_double) :: workspace(*)

  integer(c_int64_t) :: i
  real(c_double) :: t, va, vd, ve, vc

  !$omp parallel do simd
  do i = 1, len_1d
    va = a(i); vd = d(i); ve = e(i); vc = c(i)
    t = vd * ve
    b(i) = t
    a(i) = va + t * vc
  end do
end subroutine tsvc_2_s152_fp64
