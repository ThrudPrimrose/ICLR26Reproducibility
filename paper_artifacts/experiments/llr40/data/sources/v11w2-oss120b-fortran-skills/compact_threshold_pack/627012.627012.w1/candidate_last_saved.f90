subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, LEN_1D) bind(C, name="compact_threshold_pack_fp64")
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: src(LEN_1D)
    real(c_double), intent(in) :: weight(LEN_1D)
    integer(c_int64_t), intent(out) :: out_count(1)
    real(c_double), intent(inout) :: packed(LEN_1D)
    
    integer(c_int64_t) :: i, total
    ! Ensure packed array is zeroed to avoid leftover data
    do i = 1, LEN_1D
        packed(i) = 0.0_c_double
    end do
    total = 0_c_int64_t
    do i = 1, LEN_1D
        if (src(i) > 0.0_c_double) then
            total = total + 1_c_int64_t
            packed(total) = src(i) * weight(i)
        end if
    end do
    out_count(1) = total
    ! Zeroing of leftover packed entries omitted
end subroutine compact_threshold_pack_fp64
