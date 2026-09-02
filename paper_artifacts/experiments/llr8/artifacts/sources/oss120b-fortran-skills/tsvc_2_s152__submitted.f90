module tsvc_2_s152_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s152_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    integer(c_int64_t) :: i
    !$omp parallel do simd default(none) shared(a,b,c,d,e,LEN_1D) private(i)
    do i = 1, LEN_1D
      b(i) = d(i) * e(i)
      a(i) = a(i) + b(i) * c(i)
    end do
  end subroutine tsvc_2_s152_fp64
end module tsvc_2_s152_mod
