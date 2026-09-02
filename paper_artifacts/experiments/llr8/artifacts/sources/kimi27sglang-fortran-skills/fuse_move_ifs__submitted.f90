subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: K, LEN_2D, workspace_size
  type(c_ptr), value, intent(in) :: workspace
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D), b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D), src(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j
  if (K > 0) then
    !$omp parallel do private(j)
    do i = 1, LEN_2D
      if (cond(i) > 0.0d0) then
        do j = 1, LEN_2D
          a(j, i) = src(j, i) * 2.0d0
        end do
      end if
      do j = 1, LEN_2D
        b(j, i) = src(j, i) + 1.0d0
      end do
    end do
  else
    !$omp parallel do private(j)
    do i = 1, LEN_2D
      if (cond(i) > 0.0d0) then
        do j = 1, LEN_2D
          a(j, i) = src(j, i) * 2.0d0
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
