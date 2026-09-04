subroutine tsvc_2_s318_fp64(a, result, len_1d, inc) bind(c, name="tsvc_2_s318_fp64")
  use iso_c_binding
  implicit none
  real(kind=c_double),    intent(in)  :: a(*)
  real(kind=c_double),    intent(out) :: result(1)
  integer(kind=c_int64_t), intent(in) :: len_1d, inc

  integer(kind=c_int64_t) :: i, index
  real(kind=c_double)     :: v, maxv, chksum
  integer(kind=c_int64_t) :: k

  k = 0_c_int64_t
  index = 0_c_int64_t
  maxv = dabs(a(1))
  k = k + inc
  do i = 1_c_int64_t, len_1d - 1_c_int64_t
    v = dabs(a(1 + k))
    if (v > maxv) then
      index = i
      maxv = v
    end if
    k = k + inc
  end do
  chksum = maxv + dble(index)
  result(1) = chksum
end subroutine tsvc_2_s318_fp64
