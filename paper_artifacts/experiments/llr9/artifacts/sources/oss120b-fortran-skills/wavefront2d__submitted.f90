subroutine wavefront2d_fp64(a, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)

  integer(c_int64_t) :: i, j, t
  integer(c_int64_t) :: i_start, i_end

  !$omp parallel private(t, i, j, i_start, i_end)
  do t = 4, 2*LEN_2D
    i_start = max(2_c_int64_t, t - LEN_2D)
    i_end = min(LEN_2D, t - 2_c_int64_t)
    !$omp do schedule(static)
    do i = i_start, i_end
      j = t - i
      a(j, i) = 0.25d0 * (a(j, i) + a(j-1, i) + a(j, i-1) + a(j-1, i-1))
    end do
    !$omp end do
  end do
  !$omp end parallel

end subroutine wavefront2d_fp64
