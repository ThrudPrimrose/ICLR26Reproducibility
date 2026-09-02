! Optimized Fortran implementation of TSVC kernel s231
! Exposes C entry point: tsvc_2_s231_fp64
subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D, workspace, ws_bytes) bind(C, name="tsvc_2_s231_fp64")
    use iso_c_binding
    implicit none
    !--- scalar arguments (value) ---
    integer(c_int64_t), value :: LEN_2D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: ws_bytes
    !--- array arguments ---
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(in) :: bb
    !--- locals ---
    integer :: i, j
    integer(c_int64_t) :: idx

    ! Parallel outer loop over columns (i)
    !$omp parallel do schedule(static) default(none) private(i, j, idx) &
    !$omp& shared(aa, bb, LEN_2D)
    do i = 1, LEN_2D
                        do j = 2, LEN_2D
            idx = (j - 1) * LEN_2D + i
            aa(idx) = aa(idx - LEN_2D) + bb(idx)
        end do
    end do
    !$omp end parallel do

    return
end subroutine tsvc_2_s231_fp64
