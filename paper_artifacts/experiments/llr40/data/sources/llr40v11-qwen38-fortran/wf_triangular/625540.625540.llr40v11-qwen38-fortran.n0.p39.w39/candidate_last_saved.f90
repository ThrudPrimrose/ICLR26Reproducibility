subroutine wf_triangular_fp64(a, n) bind(C, name="wf_triangular_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(*)
  integer(c_int64_t) :: i, j
  do i = 2, n
    do j = i, n
      a((i-1)*n + j) = a((i-1)*n + j) + a((i-2)*n + j) + a((i-1)*n + j - 1)
    end do
  end do
end subroutine wf_triangular_fp64
