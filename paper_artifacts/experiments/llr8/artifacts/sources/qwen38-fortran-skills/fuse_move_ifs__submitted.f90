subroutine fuse_move_ifs_fp64(a, b, cond, src, k, len_2d, workspace, workspace_size) bind(C)
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: k, len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  real(c_double), intent(inout) :: b(len_2d, len_2d)
  real(c_double), intent(in)    :: cond(len_2d)
  real(c_double), intent(in)    :: src(len_2d, len_2d)
  type(c_ptr), value :: workspace

  integer(c_int64_t) :: i, j

  if (k > 0) then
    ! Fused single pass over src: b is unconditional, a only where cond>0.
    ! src read exactly once. Row i independent -> thread the outer loop.
    !$omp parallel do
    do i = 1, len_2d
      if (cond(i) > 0.0d0) then
        do j = 1, len_2d
          b(j, i) = src(j, i) + 1.0d0
          a(j, i) = src(j, i) * 2.0d0
        end do
      else
        do j = 1, len_2d
          b(j, i) = src(j, i) + 1.0d0
        end do
      end if
    end do
  else
    !$omp parallel do
    do i = 1, len_2d
      if (cond(i) > 0.0d0) then
        do j = 1, len_2d
          a(j, i) = src(j, i) * 2.0d0
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
