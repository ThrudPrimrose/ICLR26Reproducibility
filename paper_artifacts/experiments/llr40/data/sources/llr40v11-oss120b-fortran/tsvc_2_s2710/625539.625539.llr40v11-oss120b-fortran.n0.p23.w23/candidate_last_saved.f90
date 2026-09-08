module tsvc_2_s2710_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
    ! Arguments: arrays of double precision, LEN_1D elements
    real(c_double), dimension(*), intent(inout) :: a, b, c
    real(c_double), dimension(*), intent(in) :: d, e, x
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    logical :: len_gt_10, x0_pos
    real(c_double) :: ai, bi, di, ei

    len_gt_10 = LEN_1D > 10_c_int64_t
    x0_pos = x(1) > 0.0_c_double

    !$omp simd
    do i = 1, LEN_1D
      ai = a(i)
      bi = b(i)
      di = d(i)
      ei = e(i)
      if (ai > bi) then
        a(i) = ai + bi * di
        if (len_gt_10) then
          c(i) = c(i) + di * di
        else
          c(i) = di * ei + 1.0_c_double
        end if
      else
        b(i) = ai + ei * ei
        if (x0_pos) then
          c(i) = ai + di * di
        else
          c(i) = c(i) + ei * ei
        end if
      end if
    end do

  end subroutine tsvc_2_s2710_fp64
end module tsvc_2_s2710_mod
