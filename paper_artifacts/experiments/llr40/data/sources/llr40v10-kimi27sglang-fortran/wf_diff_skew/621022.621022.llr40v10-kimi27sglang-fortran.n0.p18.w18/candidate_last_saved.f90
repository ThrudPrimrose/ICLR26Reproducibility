module wf_diff_skew_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D) bind(C, name="wf_diff_skew_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(0:LEN_2D*LEN_2D-1)
    integer(c_int64_t) :: i, j, L, base, prev
    L = LEN_2D
    !$omp parallel default(private) shared(a, L)
    do i = 1, L - 1
       base = i * L
       prev = base - L
       !$omp do simd schedule(static, 1024)
       do j = 0, L - 2
          a(base + j) = a(base + j) + a(prev + j) + a(prev + j + 1)
       end do
       !$omp end do simd
    end do
    !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
