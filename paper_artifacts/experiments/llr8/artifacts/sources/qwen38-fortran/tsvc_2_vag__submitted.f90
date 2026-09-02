subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  implicit none
  real(c_double),        intent(inout) :: a(*)
  real(c_double),        intent(in)    :: b(*)
  integer(c_int32_t),    intent(in)    :: ip(*)
  integer(c_int64_t),    intent(in), value :: len_1d
  integer(c_int64_t) :: i
  integer(c_int32_t) :: idx

  !$omp parallel do schedule(static)
  do i = 1, len_1d
    idx = ip(i)
    a(i) = b(idx)
  end do
  !$omp end parallel do
end subroutine tsvc_2_vag_fp64
