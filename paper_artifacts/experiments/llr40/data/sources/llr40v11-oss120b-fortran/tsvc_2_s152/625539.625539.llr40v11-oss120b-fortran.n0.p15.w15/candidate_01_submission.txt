module tsvc_2_s152_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s152_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(out) :: b(*)
    real(c_double), intent(in) :: c(*), d(*), e(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    !$omp parallel do schedule(static) default(none) shared(a,b,c,d,e,LEN_1D) private(i)
    do i = 1_c_int64_t, LEN_1D
      b(i) = d(i) * e(i)
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s152_fp64
end module tsvc_2_s152_mod
