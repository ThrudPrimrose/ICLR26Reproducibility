subroutine wf_diff_skew_fp64(a, len_2d) bind(C)
  use, intrinsic :: iso_c_binding
  integer(c_int64_t), value :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  integer :: n, i, j
  n = int(len_2d)
  if (n > 128) then
    !$omp parallel
      do i = 2, n
        !$omp do schedule(static)
        do j = 1, n - 1
          a(j, i) = a(j, i) + a(j, i - 1) + a(j + 1, i - 1)
        end do
        !$omp end do
      end do
    !$omp end parallel
  else
    do i = 2, n
      do j = 1, n - 1
        a(j, i) = a(j, i) + a(j, i - 1) + a(j + 1, i - 1)
      end do
    end do
  end if
end subroutine wf_diff_skew_fp64
