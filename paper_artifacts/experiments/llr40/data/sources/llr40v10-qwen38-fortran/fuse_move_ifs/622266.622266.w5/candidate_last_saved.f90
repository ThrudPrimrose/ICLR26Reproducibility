subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(in)    :: cond(*)
  real(c_double), intent(in)    :: src(*)
  integer(c_int64_t), value, intent(in) :: K, LEN_2D
  integer(c_int64_t) :: i

  if (K > 0) then
    !$omp parallel do
    do i = 1, LEN_2D
      b((i-1)*LEN_2D+1:i*LEN_2D) = src((i-1)*LEN_2D+1:i*LEN_2D) + 1.0d0
      if (cond(i) > 0.0d0) then
        a((i-1)*LEN_2D+1:i*LEN_2D) = src((i-1)*LEN_2D+1:i*LEN_2D) * 2.0d0
      end if
    end do
  else
    !$omp parallel do
    do i = 1, LEN_2D
      if (cond(i) > 0.0d0) then
        a((i-1)*LEN_2D+1:i*LEN_2D) = src((i-1)*LEN_2D+1:i*LEN_2D) * 2.0d0
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
