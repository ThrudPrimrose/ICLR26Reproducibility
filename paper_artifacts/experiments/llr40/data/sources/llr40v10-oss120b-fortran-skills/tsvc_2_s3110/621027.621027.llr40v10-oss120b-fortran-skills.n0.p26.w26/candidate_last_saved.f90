subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: bb(2,2)
    integer(c_int64_t) :: i, j
    real(c_double) :: maxv, v, chksum
    integer(c_int64_t) :: xindex, yindex
    integer(c_int64_t) :: best_linear_idx, idx
    ! Compute maximum value using parallel reduction
    maxv = aa(1,1)
    !$omp parallel do reduction(max:maxv) default(none) shared(aa, LEN_2D) private(i, j, v)
    do i = 1, LEN_2D
        do j = 1, LEN_2D
            v = aa(j,i)  ! transpose to match row-major layout
            if (v > maxv) maxv = v
        end do
    end do
    !$omp end parallel do
    ! Find first occurrence (in row-major order) of the maximum value
    best_linear_idx = LEN_2D * LEN_2D
    !$omp parallel do reduction(min:best_linear_idx) default(none) shared(aa, LEN_2D, maxv) private(i, j, idx)
    do i = 1, LEN_2D
        do j = 1, LEN_2D
            if (aa(j,i) == maxv) then
                idx = (i-1) * LEN_2D + (j-1)
                if (idx < best_linear_idx) best_linear_idx = idx
            end if
        end do
    end do
    !$omp end parallel do
    xindex = best_linear_idx / LEN_2D
    yindex = best_linear_idx - xindex * LEN_2D
    chksum = maxv + dble(xindex) + dble(yindex)
    bb(1,1) = chksum
end subroutine tsvc_2_s3110_fp64
