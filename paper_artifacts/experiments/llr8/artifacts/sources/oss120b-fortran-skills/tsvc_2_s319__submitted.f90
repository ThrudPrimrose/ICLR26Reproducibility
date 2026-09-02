module tsvc_2_s319_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s319_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: sum_val

    sum_val = 0.0d0
    !$omp parallel do simd reduction(+:sum_val)
    do i = 1, LEN_1D
      a(i) = c(i) + d(i)
      sum_val = sum_val + a(i)
      b(i) = c(i) + e(i)
      sum_val = sum_val + b(i)
    end do
    ! No explicit !$omp end parallel do needed for combined construct
    b(1) = sum_val
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
