subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name='fuse_move_ifs_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(:), intent(inout) :: a, b
  real(c_double), dimension(:), intent(in) :: cond, src
  integer(c_int64_t), value, intent(in) :: K, LEN_2D

  integer :: i, n

  n = int(LEN_2D)

  if (K > 0) then
    !$omp parallel do default(none) shared(a, b, cond, src, n)
    do i = 1, n
      if (cond(i) > 0.0d0) then
        a((i - 1)*n + 1:i*n) = 2.0d0 * src((i - 1)*n + 1:i*n)
      end if
      b((i - 1)*n + 1:i*n) = src((i - 1)*n + 1:i*n) + 1.0d0
    end do
  else
    !$omp parallel do default(none) shared(a, cond, src, n)
    do i = 1, n
      if (cond(i) > 0.0d0) then
        a((i - 1)*n + 1:i*n) = 2.0d0 * src((i - 1)*n + 1:i*n)
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
