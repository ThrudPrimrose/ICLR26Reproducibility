subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)

  integer(c_int64_t) :: a, b
  real(c_double) :: cval

  if (len_2d < 9) return

  ! Both reference recurrences map to (1-based Fortran):
  !   out(a, b) = out(a, b-1) + cc(a, b),  a = 9..N, b = 9..N
  ! independent per row a; the aa and bb chains share cc(a, b) -> fused.
  !$omp parallel do
  do a = 9, len_2d
    do b = 9, len_2d
      cval = cc(a, b)
      aa(a, b) = aa(a, b - 1) + cval
      bb(a, b) = bb(a, b - 1) + cval
    end do
  end do

end subroutine tsvc_2_s2233_fp64
