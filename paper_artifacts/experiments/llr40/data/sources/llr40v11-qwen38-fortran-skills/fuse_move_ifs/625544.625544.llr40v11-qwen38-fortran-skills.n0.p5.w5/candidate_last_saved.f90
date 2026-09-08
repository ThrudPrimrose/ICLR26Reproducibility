subroutine fuse_move_ifs_fp64(a, b, cond, src, k, len_2d) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: k, len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  real(c_double), intent(inout) :: b(len_2d, len_2d)
  real(c_double), intent(in) :: cond(len_2d)
  real(c_double), intent(in) :: src(len_2d, len_2d)
  integer(c_int64_t) :: i, j
  real(c_double) :: t

  if (k > 0) then
    do i = 1, len_2d
      if (cond(i) > 0.0d0) then
        do j = 1, len_2d
          t = src(j, i)
          a(j, i) = 2.0d0 * t
          b(j, i) = t + 1.0d0
        end do
      else
        do j = 1, len_2d
          b(j, i) = src(j, i) + 1.0d0
        end do
      end if
    end do
  else
    do i = 1, len_2d
      if (cond(i) > 0.0d0) then
        do j = 1, len_2d
          a(j, i) = 2.0d0 * src(j, i)
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
