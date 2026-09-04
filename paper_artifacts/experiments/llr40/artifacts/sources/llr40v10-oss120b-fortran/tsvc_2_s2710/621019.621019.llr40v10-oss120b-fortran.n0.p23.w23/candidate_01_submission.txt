module tsvc_2_s2710_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
    ! Arguments: a, b, c are inout; d, e, x are input arrays.
    real(C_DOUBLE), intent(inout) :: a(0:*), b(0:*), c(0:*)
    real(C_DOUBLE), intent(in)    :: d(0:*), e(0:*), x(0:*)
    integer(C_INT64_T), value    :: LEN_1D
    integer(C_INT64_T) :: i
    logical :: len_gt_10
    logical :: x0_gt_0
    real(C_DOUBLE), parameter :: ONE = 1.0_C_DOUBLE
    len_gt_10 = LEN_1D > 10
    x0_gt_0 = x(0) > 0.0_C_DOUBLE

    !$omp parallel do default(none) shared(a,b,c,d,e,x,LEN_1D,len_gt_10,x0_gt_0) private(i)
    do i = 0, LEN_1D - 1
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i) * d(i)
        if (len_gt_10) then
          c(i) = c(i) + d(i) * d(i)
        else
          c(i) = d(i) * e(i) + ONE
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
end module tsvc_2_s2710_mod
