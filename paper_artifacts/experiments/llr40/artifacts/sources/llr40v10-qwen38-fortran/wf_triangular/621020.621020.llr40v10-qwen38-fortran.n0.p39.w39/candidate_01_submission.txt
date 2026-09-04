subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t) :: i, j, N
  N = LEN_2D
  do i = 2, N
     do j = i, N
        a((i-1)*N + j) = a((i-1)*N + j) + a((i-2)*N + j) + a((i-1)*N + j - 1)
     end do
  end do
end subroutine
