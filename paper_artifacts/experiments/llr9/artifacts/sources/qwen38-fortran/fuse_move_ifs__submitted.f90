subroutine fuse_move_ifs_fp64(a, b, cond, src, k, len_2d, ws, ws_bytes) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  integer(c_int64_t), value, intent(in) :: k
  integer(c_int64_t), value, intent(in) :: len_2d
  integer(c_int64_t), value, intent(in) :: ws_bytes
  type(c_ptr), value, intent(in) :: ws
  real(c_double), intent(inout) :: a(len_2d, len_2d), b(len_2d, len_2d)
  real(c_double), intent(in) :: cond(len_2d), src(len_2d, len_2d)
  integer(c_int64_t) :: i, j
  if (len_2d <= 0) return
  if (k > 0) then
     !$omp parallel do schedule(dynamic, 32)
     do i = 0, len_2d - 1
        if (cond(i + 1) > 0.0d0) then
           do j = 0, len_2d - 1
              a(j + 1, i + 1) = 2.0d0 * src(j + 1, i + 1)
           end do
        end if
        do j = 0, len_2d - 1
           b(j + 1, i + 1) = 1.0d0 + src(j + 1, i + 1)
        end do
     end do
  else
     !$omp parallel do schedule(dynamic, 32)
     do i = 0, len_2d - 1
        if (cond(i + 1) > 0.0d0) then
           do j = 0, len_2d - 1
              a(j + 1, i + 1) = 2.0d0 * src(j + 1, i + 1)
           end do
        end if
     end do
  end if
end subroutine fuse_move_ifs_fp64
