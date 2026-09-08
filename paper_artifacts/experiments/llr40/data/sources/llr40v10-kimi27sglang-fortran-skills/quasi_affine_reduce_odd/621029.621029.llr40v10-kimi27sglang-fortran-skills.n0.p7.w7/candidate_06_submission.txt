subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  real(c_double) :: acc
  integer(c_int64_t) :: k, n

  n = LEN_1D / 2
  acc = 0.0d0
  !$omp parallel do reduction(+:acc) schedule(static)
  do k = 1, n
    acc = acc + a(2*k)
  end do
  !$omp end parallel do

  out = acc
end subroutine quasi_affine_reduce_odd_fp64
