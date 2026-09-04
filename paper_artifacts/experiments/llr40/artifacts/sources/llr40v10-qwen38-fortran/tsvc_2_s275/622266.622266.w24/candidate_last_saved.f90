subroutine tsvc_2_s275_fp64(aa, bb, cc, n) bind(C, name="tsvc_2_s275_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*), cc(*)
  integer(c_int64_t), intent(in) :: n
  integer(c_int64_t) :: i, j
  integer(c_int64_t) :: active

  active = 0
  do i = 0, n - 1
    if (aa(i) > 0.0d0) active = active + 1
  end do
  write(6, '(A)') "PROBE n=" // trim(adjustl(intoa(n))) // " active=" // trim(adjustl(intoa(active)))
  flush(6)

  do i = 0, n - 1
    if (aa(i) > 0.0d0) then
      do j = 1, n - 1
        aa(j * n + i) = aa((j - 1) * n + i) + bb(j * n + i) * cc(j * n + i)
      end do
    end if
  end do

contains
  function intoa(x) result(s)
    integer(c_int64_t) :: x
    character(24) :: s
    write(s, '(i0)') x
  end function
end subroutine tsvc_2_s275_fp64
