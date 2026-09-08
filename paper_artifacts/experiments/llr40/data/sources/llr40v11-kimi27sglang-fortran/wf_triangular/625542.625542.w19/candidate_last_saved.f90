subroutine wf_triangular_fp64(a, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D*LEN_2D)
  integer(c_int64_t) :: i, j

  do i = 2, LEN_2D
    do j = i, LEN_2D
      a((i-1)*LEN_2D + j) = a((i-1)*LEN_2D + j) + a((i-2)*LEN_2D + j) + a((i-1)*LEN_2D + j - 1)
    end do
  end do
end subroutine wf_triangular_fp64
