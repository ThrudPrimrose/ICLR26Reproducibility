subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(c, name="fuse_move_ifs_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: K, LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D*LEN_2D)
  real(c_double), intent(inout) :: b(LEN_2D*LEN_2D)
  real(c_double), intent(in) :: cond(LEN_2D)
  real(c_double), intent(in) :: src(LEN_2D*LEN_2D)
  integer(c_int64_t) :: i, j, base

  if (K > 0_c_int64_t) then
    !$omp parallel do private(i,j,base) schedule(static)
    do i = 1_c_int64_t, LEN_2D
      base = (i - 1_c_int64_t) * LEN_2D
      if (cond(i) > 0.0_c_double) then
        !$omp simd nontemporal(a)
        do j = 1_c_int64_t, LEN_2D
          a(base + j) = src(base + j) * 2.0_c_double
        end do
        !$omp end simd
      end if
      !$omp simd nontemporal(b)
      do j = 1_c_int64_t, LEN_2D
        b(base + j) = src(base + j) + 1.0_c_double
      end do
      !$omp end simd
    end do
    !$omp end parallel do
  else
    !$omp parallel do private(i,j,base) schedule(static)
    do i = 1_c_int64_t, LEN_2D
      base = (i - 1_c_int64_t) * LEN_2D
      if (cond(i) > 0.0_c_double) then
        !$omp simd nontemporal(a)
        do j = 1_c_int64_t, LEN_2D
          a(base + j) = src(base + j) * 2.0_c_double
        end do
        !$omp end simd
      end if
    end do
    !$omp end parallel do
  end if
end subroutine fuse_move_ifs_fp64
