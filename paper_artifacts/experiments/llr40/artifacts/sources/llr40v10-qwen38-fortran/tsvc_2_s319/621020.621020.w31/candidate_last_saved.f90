subroutine tsvc_2_s319_fp64(a, b, c, d, e, n) bind(C, name="tsvc_2_s319_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), intent(inout) :: a, b
  type(c_ptr), intent(in) :: c, d, e
  integer(c_int64_t), intent(in) :: n
  real(c_double), dimension(:), pointer :: pa, pb, pc, pd, pe
  integer(c_int64_t) :: i
  real(c_double) :: s

  call c_f_pointer(a, pa, [n])
  call c_f_pointer(b, pb, [n])
  call c_f_pointer(c, pc, [n])
  call c_f_pointer(d, pd, [n])
  call c_f_pointer(e, pe, [n])

  s = 0.0d0
  do i = 1, n
    pa(i) = pc(i) + pd(i)
    s = s + pa(i)
    pb(i) = pc(i) + pe(i)
    s = s + pb(i)
  end do
  if (n > 0) pb(1) = s
end subroutine tsvc_2_s319_fp64
