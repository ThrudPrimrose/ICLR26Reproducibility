subroutine wf_triangular_fp64(a, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: n, d, r, c, rmin, rmax

  n = LEN_2D
  if (n < 2) return

  !$omp parallel private(r, c, rmin, rmax)
  do d = 4, 2 * n
    rmin = max(2_c_int64_t, d - n)
    rmax = min(d / 2, n)
    !$omp do
    do r = rmin, rmax
      c = d - r
      a(c, r) = a(c, r) + a(c, r - 1) + a(c - 1, r)
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wf_triangular_fp64
