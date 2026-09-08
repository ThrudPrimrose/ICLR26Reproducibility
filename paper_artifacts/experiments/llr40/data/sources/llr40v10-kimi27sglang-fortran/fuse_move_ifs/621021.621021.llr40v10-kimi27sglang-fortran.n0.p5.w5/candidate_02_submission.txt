subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(c, name='fuse_move_ifs_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D), b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D), src(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  if (K > 0) then
    !$omp parallel do private(i, j) schedule(static)
    do i = 1, LEN_2D
      if (cond(i) > 0.0_c_double) then
        !$omp simd
        do j = 1, LEN_2D
          a(j, i) = src(j, i) * 2.0_c_double
        end do
      end if
      !$omp simd
      do j = 1, LEN_2D
        b(j, i) = src(j, i) + 1.0_c_double
      end do
    end do
    !$omp end parallel do
  else
    !$omp parallel do private(i, j) schedule(static)
    do i = 1, LEN_2D
      if (cond(i) > 0.0_c_double) then
        !$omp simd
        do j = 1, LEN_2D
          a(j, i) = src(j, i) * 2.0_c_double
        end do
      end if
    end do
    !$omp end parallel do
  end if
end subroutine fuse_move_ifs_fp64
