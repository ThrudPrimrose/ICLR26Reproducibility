module wf_diff_skew_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D, workspace, workspace_size) bind(C)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i, j

    !$omp parallel default(none) shared(a, LEN_2D) private(i, j)
    do i = 2, LEN_2D
      !$omp do schedule(static)
      do j = 1, LEN_2D-1
        a(j, i) = a(j, i) + a(j, i-1) + a(j+1, i-1)
      end do
      !$omp end do
    end do
    !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
