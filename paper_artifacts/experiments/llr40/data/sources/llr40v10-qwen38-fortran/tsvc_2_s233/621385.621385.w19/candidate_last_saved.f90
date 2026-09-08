subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C, name='tsvc_2_s233_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*), bb(*)
  real(c_double), intent(in) :: cc(*)
  integer(c_int64_t) :: len_2d
  integer(c_int64_t) :: i, j, n
  real(c_double) :: s

  n = len_2d
  !$omp parallel
  !$omp do schedule(static)
  do i = 8, n-1
    s = aa(7*n + i)
    do j = 8, n-1
      s = s + cc(j*n + i)
      aa(j*n + i) = s
    end do
  end do
  !$omp do schedule(static)
  do j = 8, n-1
    s = bb(j*n + 7)
    do i = 8, n-1
      s = s + cc(j*n + i)
      bb(j*n + i) = s
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s233_fp64
