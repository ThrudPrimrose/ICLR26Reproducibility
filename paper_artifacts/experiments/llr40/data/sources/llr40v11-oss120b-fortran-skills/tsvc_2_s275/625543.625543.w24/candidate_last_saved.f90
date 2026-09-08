module tsvc_2_s275_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
    ! Arguments: aa - inout, bb - in, cc - in, LEN_2D - size (scalar)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j
    real(c_double) :: acc

    do i = 1, LEN_2D
      if (aa(i, 1) > 0.0d0) then
        acc = aa(i, 1)
        do j = 2, LEN_2D
          acc = acc + bb(i, j) * cc(i, j)
          aa(i, j) = acc
        end do
      end if
    end do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
