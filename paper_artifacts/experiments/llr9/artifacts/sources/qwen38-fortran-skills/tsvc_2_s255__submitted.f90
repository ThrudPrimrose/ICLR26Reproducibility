subroutine tsvc_2_s255_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, n

  n = len_1d
  if (n == 0) return
  if (workspace_size > 0) workspace(1) = 0.0d0
  if (n == 1) then
    ! numpy b[-1] wraps to b[0]; x = y = b(1)
    a(1) = (b(1) + b(1) + b(1)) * 0.333d0
    return
  end if
  ! wrap-around head elements (exactly the reference's (b[i] + x) + y order)
  a(1) = (b(1) + b(n) + b(n - 1)) * 0.333d0
  a(2) = (b(2) + b(1) + b(n)) * 0.333d0
  ! bulk: rotated scalars are a false dependence -> pointwise stencil,
  ! no axis carries a dependence; unit stride vectorizes, threads fill cores
  !$omp parallel do simd
  do i = 3, n
    a(i) = (b(i) + b(i - 1) + b(i - 2)) * 0.333d0
  end do
end subroutine tsvc_2_s255_fp64
