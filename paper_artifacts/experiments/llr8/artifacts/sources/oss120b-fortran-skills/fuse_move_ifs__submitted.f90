subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C)
  use iso_c_binding
  implicit none

  integer(c_int64_t), value, intent(in) :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D)
  real(c_double), intent(in) :: src(LEN_2D, LEN_2D)

  integer(c_int64_t) :: i, j

  !$omp parallel do private(j) schedule(static)
  do i = 1, LEN_2D
    !$omp simd
    do j = 1, LEN_2D
      if (cond(i) > 0.0_c_double) then
        a(j, i) = src(j, i) * 2.0_c_double
      end if
      if (K > 0) then
        b(j, i) = src(j, i) + 1.0_c_double
      end if
    end do
  end do
  !$omp end parallel do

end subroutine fuse_move_ifs_fp64
