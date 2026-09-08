subroutine tsvc_2_s3111_fp64(a, b, len1d) bind(C, name="tsvc_2_s3111_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value :: b
  integer(c_int64_t), value, intent(in) :: len1d
  real(c_double), pointer :: arr(:), out(:)
  real(c_double) :: s
  integer(c_int64_t) :: i, n

  n = len1d
  call c_f_pointer(a, arr, [n])
  call c_f_pointer(b, out, [1])
  s = 0.0d0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    if (arr(i) > 0.0d0) s = s + arr(i)
  end do
  out(1) = s
end subroutine tsvc_2_s3111_fp64
