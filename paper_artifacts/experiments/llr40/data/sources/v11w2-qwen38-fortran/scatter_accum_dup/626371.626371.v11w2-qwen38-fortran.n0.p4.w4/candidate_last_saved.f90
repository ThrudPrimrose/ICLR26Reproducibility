subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: bins(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    integer(c_int64_t) :: i
    if (LEN_1D < 262144_8) then
        do i = 1, LEN_1D
            bins(ip(i)) = bins(ip(i)) + src(i)
        end do
    else
        !$omp parallel do
        do i = 1, LEN_1D
            !$omp atomic update
            bins(ip(i)) = bins(ip(i)) + src(i)
        end do
        !$omp end parallel do
    end if
end subroutine scatter_accum_dup_fp64
