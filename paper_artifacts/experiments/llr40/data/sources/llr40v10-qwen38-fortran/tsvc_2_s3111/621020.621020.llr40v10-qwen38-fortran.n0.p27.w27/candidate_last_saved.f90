subroutine tsvc_2_s3111_fp64(a, b, len_1d) bind(c, name='tsvc_2_s3111_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(in)    :: a(*)
  real(c_double), intent(out)   :: b(*)
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double) :: s
  integer(c_int64_t) :: i
  integer(kind=8) :: cyc0, cyc1
  integer :: callno
  save callno

  interface
    function rdtsc() result(r) bind(c, name='__rdtsc')
      integer(kind=8) :: r
    end function
  end interface

  callno = callno + 1
  cyc0 = rdtsc()
  s = 0.0d0
  do i = 1, len_1d
     if (a(i) > 0.0d0) s = s + a(i)
  end do
  b(1) = s
  cyc1 = rdtsc()
  if (callno <= 3 .or. modulo(callno, 5) == 1) then
     write(*,'(A,I0,A,I0,A,I0)') ' call=', callno, ' n=', len_1d, ' cycles=', (cyc1 - cyc0)
     flush(0)
  end if
end subroutine tsvc_2_s3111_fp64
