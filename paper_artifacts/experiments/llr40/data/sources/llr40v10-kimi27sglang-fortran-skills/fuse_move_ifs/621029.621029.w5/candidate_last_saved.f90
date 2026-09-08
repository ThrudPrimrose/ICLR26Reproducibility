subroutine fuse_move_ifs(a, b, src, cond, K, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: src(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D)
  integer(c_int64_t) :: i, j

  if (K > 0_c_int64_t) then
    !$omp parallel do
    do i = 1_c_int64_t, LEN_2D
      if (cond(i) > 0.0d0) then
        !$omp simd
        do j = 1_c_int64_t, LEN_2D
          a(j, i) = src(j, i) * 2.0d0
          b(j, i) = src(j, i) + 1.0d0
        end do
      else
        !$omp simd
        do j = 1_c_int64_t, LEN_2D
          b(j, i) = src(j, i) + 1.0d0
        end do
      end if
    end do
  else
    !$omp parallel do
    do i = 1_c_int64_t, LEN_2D
      if (cond(i) > 0.0d0) then
        !$omp simd
        do j = 1_c_int64_t, LEN_2D
          a(j, i) = src(j, i) * 2.0d0
        end do
      end if
    end do
  end if
end subroutine fuse_move_ifs
