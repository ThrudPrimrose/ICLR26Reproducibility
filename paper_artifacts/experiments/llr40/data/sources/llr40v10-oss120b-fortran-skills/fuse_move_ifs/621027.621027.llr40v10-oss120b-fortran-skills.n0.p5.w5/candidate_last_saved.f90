module fuse_move_ifs_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in)    :: cond(*)
    real(c_double), intent(in)    :: src(*)
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t) :: i, j, idx
    if (K > 0_c_int64_t) then
      !$omp parallel do schedule(static) default(none) shared(a,b,cond,src,LEN_2D,K) private(idx,j)
      do i = 1_c_int64_t, LEN_2D
        !$omp simd
        do j = 1_c_int64_t, LEN_2D
          idx = (i - 1_c_int64_t) * LEN_2D + j
          b(idx) = src(idx) + 1.0d0
          if (cond(i) > 0.0d0) then
            a(idx) = src(idx) * 2.0d0
          end if
        end do
      end do
      !$omp end parallel do
    else
      !$omp parallel do schedule(static) default(none) shared(a,cond,src,LEN_2D) private(idx,j)
      do i = 1_c_int64_t, LEN_2D
        if (cond(i) > 0.0d0) then
          !$omp simd
          do j = 1_c_int64_t, LEN_2D
            idx = (i - 1_c_int64_t) * LEN_2D + j
            a(idx) = src(idx) * 2.0d0
          end do
        end if
      end do
      !$omp end parallel do
    end if
  end subroutine fuse_move_ifs_fp64
end module fuse_move_ifs_mod
