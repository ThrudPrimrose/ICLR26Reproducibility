subroutine tsvc_2_s13110_fp64(aa, bb, len_2d, ws, ws_bytes) bind(C)
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: aa(*)
  real(c_double), intent(out) :: bb(2,2)
  integer(c_int64_t), value, intent(in) :: len_2d
  type(c_ptr), intent(in) :: ws
  integer(c_int64_t), value, intent(in) :: ws_bytes

  real(c_double) :: maxv, chksum
  integer(c_int64_t) :: n, nn, i, wi

  n = len_2d
  nn = n * n
  maxv = aa(1)
  wi = 0
  do i = 2, nn
    if (aa(i) > maxv) then
      maxv = aa(i)
      wi = i - 1
    end if
  end do
  chksum = maxv + dble(wi / n) + dble(mod(wi, n))
  bb(1,1) = chksum
end subroutine tsvc_2_s13110_fp64
