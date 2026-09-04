module fuse_move_ifs_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
    ! Arguments: a, b, cond, src are double* (C double), K and LEN_2D are int64_t passed by value.
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in)    :: cond(*)
    real(c_double), intent(in)    :: src(*)
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j, idx
    ! Parallelize over rows (i). Each row is independent.
    !$omp parallel do schedule(static) private(i, j, idx)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      ! Compute a for rows where cond > 0
      if (cond(i+1) > 0.0_c_double) then
        !$omp simd
        do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
          idx = i * LEN_2D + j + 1_c_int64_t
          a(idx) = src(idx) * 2.0_c_double
        end do
      end if
      ! Compute b for all rows if K > 0
      if (K > 0_c_int64_t) then
        !$omp simd
        do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
          idx = i * LEN_2D + j + 1_c_int64_t
          b(idx) = src(idx) + 1.0_c_double
        end do
      end if
    end do
    !$omp end parallel do
  end subroutine fuse_move_ifs_fp64
end module fuse_move_ifs_mod
