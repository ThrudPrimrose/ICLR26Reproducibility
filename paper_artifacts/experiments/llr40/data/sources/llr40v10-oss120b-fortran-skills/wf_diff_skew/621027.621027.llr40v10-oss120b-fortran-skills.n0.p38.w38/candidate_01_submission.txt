module wf_diff_skew_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_diff_skew_fp64(a, LEN_2D, workspace, workspace_size) bind(C, name="wf_diff_skew_fp64")
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D*LEN_2D)
    type(c_ptr), value :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i, j
  !$omp parallel default(none) shared(a, LEN_2D) private(i,j)
    do i = 2, LEN_2D
      !$omp do
      do j = 1, LEN_2D - 1
        a((i-1)*LEN_2D + j) = a((i-1)*LEN_2D + j) + a((i-2)*LEN_2D + j) + a((i-2)*LEN_2D + (j+1))
      end do
      !$omp end do
    end do
  !$omp end parallel
  end subroutine wf_diff_skew_fp64
end module wf_diff_skew_mod
