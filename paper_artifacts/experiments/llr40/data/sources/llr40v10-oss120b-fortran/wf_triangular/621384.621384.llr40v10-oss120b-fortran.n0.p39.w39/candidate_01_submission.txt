module wf_triangular_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
    implicit none
    real(c_double), intent(inout) :: a(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: s, i, j, i_start, i_end
    ! Parallel wavefront using anti-diagonal decomposition
    !$omp parallel default(none) shared(a, LEN_2D) private(s,i_start,i_end,i,j)
    do s = 2_c_int64_t, 2_c_int64_t*LEN_2D - 2_c_int64_t
      i_start = max(1_c_int64_t, s - (LEN_2D - 1_c_int64_t))
      i_end   = min(LEN_2D - 1_c_int64_t, s / 2_c_int64_t)
      !$omp do simd
      do i = i_start, i_end
        j = s - i
        a(i*LEN_2D + j + 1) = a(i*LEN_2D + j + 1) + a((i-1)*LEN_2D + j + 1) + a(i*LEN_2D + j)
      end do
      !$omp end do simd
    end do
    !$omp end parallel
  end subroutine wf_triangular_fp64
end module wf_triangular_mod
