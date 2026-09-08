subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
  real(c_double), intent(in)    :: cond(LEN_2D)
  real(c_double), intent(in)    :: src(LEN_2D, LEN_2D)

  integer(kind=c_int64_t) :: i

  if (K > 0) then
!$omp parallel do default(none) shared(a, b, cond, src, LEN_2D)
    do i = 1, LEN_2D
       if (cond(i) > 0.0d0) then
          a(:, i) = 2.0d0 * src(:, i)
       end if
       b(:, i) = src(:, i) + 1.0d0
    end do
!$omp end parallel do
  else
!$omp parallel do default(none) shared(a, cond, src, LEN_2D)
    do i = 1, LEN_2D
       if (cond(i) > 0.0d0) then
          a(:, i) = 2.0d0 * src(:, i)
       end if
    end do
!$omp end parallel do
  end if
end subroutine fuse_move_ifs_fp64
