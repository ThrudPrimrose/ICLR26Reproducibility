subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s152_fp64")
  use iso_c_binding
  implicit none
  real(C_DOUBLE), intent(inout) :: a(0:*)
  real(C_DOUBLE), intent(out)   :: b(0:*)
  real(C_DOUBLE), intent(in)    :: c(0:*), d(0:*), e(0:*)
  integer(C_INT64_T), value    :: LEN_1D
  integer(C_INT64_T) :: i

  if (LEN_1D > 1000) then
    !$omp parallel do default(none) shared(a,b,c,d,e,LEN_1D) private(i) schedule(static)
    do i = 0, LEN_1D - 1
      b(i) = d(i) * e(i)
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do
  else
    do i = 0, LEN_1D - 1
      b(i) = d(i) * e(i)
      a(i) = a(i) + b(i) * c(i)
    end do
  end if
end subroutine tsvc_2_s152_fp64
