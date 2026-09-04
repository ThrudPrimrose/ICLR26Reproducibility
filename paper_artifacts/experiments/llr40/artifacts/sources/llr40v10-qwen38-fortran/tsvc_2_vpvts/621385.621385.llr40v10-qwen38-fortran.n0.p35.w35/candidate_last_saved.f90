subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding, only: c_int64_t
  implicit none
  double precision, intent(in)     :: b(*)
  integer(c_int64_t), intent(in), value :: len_1d, s
  double precision, intent(inout)  :: a(*)
  double precision :: ss
  integer(c_int64_t) :: i
  integer, parameter :: npar = 1000000
  ss = real(s, kind=8)
  if (len_1d > npar) then
    !$omp parallel do
    do i = 1, len_1d
      a(i) = a(i) + b(i) * ss
    end do
  else
    do i = 1, len_1d
      a(i) = a(i) + b(i) * ss
    end do
  end if
end subroutine
