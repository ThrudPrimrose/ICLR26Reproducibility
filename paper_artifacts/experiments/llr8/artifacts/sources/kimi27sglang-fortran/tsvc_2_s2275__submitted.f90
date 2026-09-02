subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(c)
  use iso_c_binding, only: c_double, c_int64_t
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D), aa(LEN_2D,LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D), bb(LEN_2D,LEN_2D), c(LEN_2D), cc(LEN_2D,LEN_2D), d(LEN_2D)
  integer(c_int64_t) :: i, j
  do i = 1_c_int64_t, LEN_2D
    do j = 1_c_int64_t, LEN_2D
      aa(j,i) = aa(j,i) + bb(j,i) * cc(j,i)
    end do
    a(i) = b(i) + c(i) * d(i)
  end do
end subroutine tsvc_2_s2275_fp64
