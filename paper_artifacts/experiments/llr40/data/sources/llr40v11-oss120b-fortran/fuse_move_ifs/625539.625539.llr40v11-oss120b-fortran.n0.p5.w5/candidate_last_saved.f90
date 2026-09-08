module fuse_move_ifs_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
    ! Arguments correspond to C signature: double* a, double* b, const double* cond, const double* src, int64_t K, int64_t LEN_2D
    implicit none
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_2D
    real(c_double), dimension(*), intent(inout) :: a
    real(c_double), dimension(*), intent(inout) :: b
    real(c_double), dimension(*), intent(in) :: src
    real(c_double), dimension(*), intent(in) :: cond
    integer(c_int64_t) :: i, base_idx, idx
    !$omp parallel do schedule(static) private(i, base_idx, idx)
    do i = 1, LEN_2D
        base_idx = (i - 1) * LEN_2D
        if (K > 0_c_int64_t) then
            !$omp simd
            do idx = base_idx + 1, base_idx + LEN_2D
                b(idx) = src(idx) + 1.0_c_double
            end do
        end if
        if (cond(i) > 0.0_c_double) then
            !$omp simd
            do idx = base_idx + 1, base_idx + LEN_2D
                a(idx) = src(idx) * 2.0_c_double
            end do
        end if
    end do
    !$omp end parallel do
  end subroutine fuse_move_ifs_fp64
end module fuse_move_ifs_mod
