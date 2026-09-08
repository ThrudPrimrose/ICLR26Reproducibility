subroutine fuse_move_ifs_fp64(a, b, cond, src, k, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: k, len_2d
  real(c_double), intent(inout) :: a(len_2d*len_2d)
  real(c_double), intent(inout) :: b(len_2d*len_2d)
  real(c_double), intent(in) :: cond(len_2d)
  real(c_double), intent(in) :: src(len_2d*len_2d)
  integer(c_int64_t) :: i, j, l2

  l2 = len_2d
  if (l2 <= 0) return

  ! C layout: element (i,j) 0-based lives at flat i*LEN_2D + j; a C row is a
  ! contiguous block. Both nests carry no dependence, so thread on i and let
  ! the unit-stride inner loop vectorize.  When k > 0 the two nests read the
  ! same src elements, so fuse them: one pass over src instead of two.
  if (k > 0) then
    !$omp parallel do schedule(static)
    do i = 0, l2 - 1
      if (cond(i + 1) > 0.0d0) then
        !$omp simd
        do j = 1, l2
          b(i*l2 + j) = src(i*l2 + j) + 1.0d0
          a(i*l2 + j) = src(i*l2 + j) * 2.0d0
        end do
      else
        !$omp simd
        do j = 1, l2
          b(i*l2 + j) = src(i*l2 + j) + 1.0d0
        end do
      end if
    end do
  else
    !$omp parallel do schedule(static)
    do i = 0, l2 - 1
      if (cond(i + 1) > 0.0d0) then
        !$omp simd
        do j = 1, l2
          a(i*l2 + j) = src(i*l2 + j) * 2.0d0
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs_fp64
