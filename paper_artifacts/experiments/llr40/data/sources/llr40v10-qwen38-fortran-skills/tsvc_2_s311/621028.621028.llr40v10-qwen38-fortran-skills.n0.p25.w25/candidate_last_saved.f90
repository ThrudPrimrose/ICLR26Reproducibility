subroutine tsvc_2_s311_fp64(a, sum_out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: sum_out(len_1d)
  real(c_double) :: s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15, s
  integer(c_int64_t) :: i, n16

  n16 = (len_1d / 16) * 16
  s0=0.0d0; s1=0.0d0; s2=0.0d0; s3=0.0d0; s4=0.0d0; s5=0.0d0; s6=0.0d0; s7=0.0d0
  s8=0.0d0; s9=0.0d0; s10=0.0d0; s11=0.0d0; s12=0.0d0; s13=0.0d0; s14=0.0d0; s15=0.0d0
  !$omp parallel do schedule(static) reduction(+:s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15)
  do i = 1, n16, 16
    s0 = s0 + a(i)
    s1 = s1 + a(i+1)
    s2 = s2 + a(i+2)
    s3 = s3 + a(i+3)
    s4 = s4 + a(i+4)
    s5 = s5 + a(i+5)
    s6 = s6 + a(i+6)
    s7 = s7 + a(i+7)
    s8 = s8 + a(i+8)
    s9 = s9 + a(i+9)
    s10 = s10 + a(i+10)
    s11 = s11 + a(i+11)
    s12 = s12 + a(i+12)
    s13 = s13 + a(i+13)
    s14 = s14 + a(i+14)
    s15 = s15 + a(i+15)
  end do
  do i = n16 + 1, len_1d
    s0 = s0 + a(i)
  end do
  s = s0+s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11+s12+s13+s14+s15
  sum_out(1) = s
end subroutine tsvc_2_s311_fp64
