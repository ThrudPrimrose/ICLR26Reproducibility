subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: N
  integer(c_int64_t) :: t, i, j, j_lower, j_upper, tmp_lower
  N = LEN_2D
  ! Parallel wavefront over anti-diagonals (i + j constant)
  !$omp parallel private(t,j,i,j_lower,j_upper,tmp_lower) shared(a,N)
  do t = 4, 2*N
    tmp_lower = max(t - N, (t + 1_c_int64_t) / 2_c_int64_t)
    j_lower = max(2_c_int64_t, tmp_lower)
    j_upper = min(N, t - 2_c_int64_t)
    !$omp do schedule(static) private(j,i)
    do j = j_lower, j_upper
      i = t - j
      a(j, i) = a(j, i) + a(j-1, i) + a(j, i-1)
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wf_triangular_fp64
