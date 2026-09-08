subroutine tsvc_2_s252(a, b, c, len_1d) bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  integer(c_int64_t) :: n
  n = len_1d
  if (n <= 0) return
  a(1) = b(1) * c(1)
  if (n > 1) then
    a(2:n) = b(2:n) * c(2:n) + b(1:n-1) * c(1:n-1)
  end if
end subroutine tsvc_2_s252
