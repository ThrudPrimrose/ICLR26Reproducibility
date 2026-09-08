subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d) bind(C, name='tsvc_2_s275_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in) :: bb(*), cc(*)
  integer(c_int64_t) :: n, i, j, off
  real(c_double) :: s

  n = len_2d

!$omp parallel do schedule(static) private(s, off)
  do i = 0, n - 1
    if (aa(i + 1) > 0.0d0) then
      s = aa(i + 1)
      off = i + 1 + n
      do j = 1, n - 1
        s = s + bb(off) * cc(off)
        aa(off) = s
        off = off + n
      end do
    end if
  end do
!$omp end parallel do
end subroutine tsvc_2_s275_fp64
