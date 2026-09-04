module tsvc_2_s2710_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(inout) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    real(c_double), intent(in) :: x(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    !$omp simd
    do i = 1, LEN_1D
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i) * d(i)
        if (LEN_1D > 10_c_int64_t) then
          c(i) = c(i) + d(i) * d(i)
        else
          c(i) = d(i) * e(i) + 1.0_c_double
        end if
      else
        b(i) = a(i) + e(i) * e(i)
        if (x(1) > 0.0_c_double) then
          c(i) = a(i) + d(i) * d(i)
        else
          c(i) = c(i) + e(i) * e(i)
        end if
      end if
    end do
  end subroutine tsvc_2_s2710_fp64
end module tsvc_2_s2710_mod
