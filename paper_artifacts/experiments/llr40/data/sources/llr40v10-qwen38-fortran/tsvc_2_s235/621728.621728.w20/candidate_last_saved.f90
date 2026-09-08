subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in) :: b(*)
  real(c_double), intent(in) :: bb(*)
  real(c_double), intent(in) :: c(*)
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t) :: i, j, N, NN
  N = LEN_2D
  NN = N * N
  do i = 0, N-1
    a(i+1) = a(i+1) + b(i+1) * c(i+1)
    do j = 1, N-1
      aa(j*N + i + 1) = aa((j-1)*N + i + 1) + bb(j*N + i + 1) * a(i+1)
    end do
  end do
end subroutine
