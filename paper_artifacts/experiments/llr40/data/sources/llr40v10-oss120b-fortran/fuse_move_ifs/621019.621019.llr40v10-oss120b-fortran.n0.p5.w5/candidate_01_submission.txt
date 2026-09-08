module fuse_move_ifs_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
    implicit none
    ! Arguments
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(out) :: b(*)
    real(c_double), intent(in) :: cond(*)
    real(c_double), intent(in) :: src(*)
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_2D
    ! Loop indices
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx
    ! First loop nest with conditional
    !$omp parallel do default(none) private(i,j,idx) shared(a,b,cond,src,LEN_2D)
    do i = 0, LEN_2D - 1
      if (cond(i+1) > 0.0_c_double) then
        !$omp simd
        do j = 0, LEN_2D - 1
          idx = i * LEN_2D + j
          a(idx+1) = src(idx+1) * 2.0_c_double
        end do
      end if
    end do
    !$omp end parallel do

    ! Second loop nest conditional on K
    if (K > 0_c_int64_t) then
      !$omp parallel do default(none) private(i,j,idx) shared(a,b,cond,src,LEN_2D)
      do i = 0, LEN_2D - 1
        !$omp simd
        do j = 0, LEN_2D - 1
          idx = i * LEN_2D + j
          b(idx+1) = src(idx+1) + 1.0_c_double
        end do
      end do
      !$omp end parallel do
    end if
  end subroutine fuse_move_ifs_fp64
end module fuse_move_ifs_mod
