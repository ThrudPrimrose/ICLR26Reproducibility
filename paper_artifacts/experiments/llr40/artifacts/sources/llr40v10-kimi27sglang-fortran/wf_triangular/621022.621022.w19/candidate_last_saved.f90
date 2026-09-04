subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name='wf_triangular_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(0:*)
  integer(c_int64_t), value :: LEN_2D
  integer(c_int64_t) :: i, j, n

  n = LEN_2D
  do i = 1, n - 1
    do j = i, n - 1
      a(i*n + j) = a(i*n + j) + a((i - 1)*n + j) + a(i*n + j - 1)
    end do
  end do
end subroutine wf_triangular_fp64
