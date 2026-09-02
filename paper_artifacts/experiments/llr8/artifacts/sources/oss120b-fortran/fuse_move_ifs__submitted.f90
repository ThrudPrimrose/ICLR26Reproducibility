subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D), b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D), src(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j
  logical :: cond_i

  if (K > 0) then
    !$omp parallel do private(i, j, cond_i) schedule(static)
    do i = 1, LEN_2D
      cond_i = cond(i) > 0.0d0
      if (cond_i) then
        !$omp simd
        do j = 1, LEN_2D
          a(j,i) = src(j,i) * 2.0d0
          b(j,i) = src(j,i) + 1.0d0
        end do
      else
        !$omp simd
        do j = 1, LEN_2D
          b(j,i) = src(j,i) + 1.0d0
        end do
      end if
    end do
  else
    !$omp parallel do private(i, j) schedule(static)
    do i = 1, LEN_2D
      if (cond(i) > 0.0d0) then
        !$omp simd
        do j = 1, LEN_2D
          a(j,i) = src(j,i) * 2.0d0
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
