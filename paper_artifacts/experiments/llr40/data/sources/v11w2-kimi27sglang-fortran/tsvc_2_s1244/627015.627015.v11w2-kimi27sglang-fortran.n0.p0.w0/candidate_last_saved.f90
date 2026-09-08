module tsvc_2_s1244_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(c, name="tsvc_2_s1244_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(in)    :: c(*)
    real(c_double), intent(inout) :: d(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, n
    real(c_double) :: bi, ci, anext, anew
    n = LEN_1D - 1
    !$omp simd safelen(8)
    do i = 1, n
      anext = a(i + 1)
      bi = b(i)
      ci = c(i)
      anew = bi + ci*ci + bi*bi + ci
      a(i) = anew
      d(i) = anew + anext
    end do
    !$omp end simd
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_mod
