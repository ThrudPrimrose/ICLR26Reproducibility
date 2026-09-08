subroutine wf_triangular_fp64(a, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: d, i, i_start, i_end, j

  !$omp parallel private(d,i,i_start,i_end,j)
  do d = 4_c_int64_t, 2_c_int64_t * LEN_2D
    i_start = max(2_c_int64_t, d - LEN_2D)
    i_end = min(LEN_2D, d / 2_c_int64_t)
    !$omp do schedule(static)
    do i = i_start, i_end
      j = d - i
      a(j,i) = a(j,i) + a(j,i-1) + a(j-1,i)
    end do
    !$omp end do
  end do
  !$omp end parallel

end subroutine wf_triangular_fp64
