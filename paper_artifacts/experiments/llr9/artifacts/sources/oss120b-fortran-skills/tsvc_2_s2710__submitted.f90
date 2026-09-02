subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s2710_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D), d(LEN_1D), e(LEN_1D)
  real(c_double), intent(in) :: x(LEN_1D)
  integer(c_int64_t) :: i
  logical :: len_gt_10, x0_gt_0

  len_gt_10 = LEN_1D > 10
  x0_gt_0 = x(1) > 0.0d0

  !$omp parallel do default(none) shared(a, b, c, d, e, x, len_gt_10, x0_gt_0, LEN_1D) private(i)
  do i = 1, LEN_1D
    if (a(i) > b(i)) then
      a(i) = a(i) + b(i) * d(i)
      if (len_gt_10) then
        c(i) = c(i) + d(i) * d(i)
      else
        c(i) = d(i) * e(i) + 1.0d0
      end if
    else
      b(i) = a(i) + e(i) * e(i)
      if (x0_gt_0) then
        c(i) = a(i) + d(i) * d(i)
      else
        c(i) = c(i) + e(i) * e(i)
      end if
    end if
  end do
  !$omp end parallel do

end subroutine tsvc_2_s2710_fp64
