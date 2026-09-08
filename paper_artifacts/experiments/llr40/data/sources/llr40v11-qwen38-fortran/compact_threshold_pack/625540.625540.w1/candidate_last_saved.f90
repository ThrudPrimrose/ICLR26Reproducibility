module ctp_mod
    use iso_c_binding
    use omp_lib
    implicit none

    integer(c_int64_t), allocatable :: counts(:), base_off(:)

contains

    subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, len_1d, &
                                           workspace, ws_bytes) &
        bind(C, name='compact_threshold_pack_fp64')
        integer(c_int64_t), intent(inout) :: out_count(1)
        real(c_double), intent(in), dimension(*) :: src
        real(c_double), intent(in), dimension(*) :: weight
        real(c_double), intent(inout), dimension(*) :: packed
        integer(c_int64_t), intent(in), value :: len_1d
        type(c_ptr), intent(in) :: workspace
        integer(c_int64_t), intent(in), value :: ws_bytes

        integer(c_int64_t) :: nch, cl, k, a, b, cnt, lc, b0, i
        integer :: nt

        if (len_1d <= 0) then
            out_count(1) = 0
            return
        end if

        nt = omp_get_max_threads()
        nch = nt * 16
        if (nch < 1) nch = 1
        if (nch > 131072) nch = 131072
        nch = min(nch, len_1d / 32)
        if (nch < 1) nch = 1

        if (allocated(counts)) then
            if (size(counts) < nch) then
                deallocate(counts, base_off)
            end if
        end if
        if (.not. allocated(counts)) then
            allocate(counts(nch), base_off(nch + 1))
        end if

        cl = (len_1d + nch - 1) / nch

        ! pass 1: per-chunk survivor counts
        !$omp parallel do
        do k = 1, nch
            a = (k - 1) * cl + 1
            b = min(k * cl, len_1d)
            cnt = 0
            do i = a, b
                if (src(i) > 0.0d0) cnt = cnt + 1
            end do
            counts(k) = cnt
        end do
        !$omp end parallel do

        ! exclusive scan of chunk counts
        base_off(1) = 0
        do k = 1, nch
            base_off(k + 1) = base_off(k) + counts(k)
        end do
        out_count(1) = base_off(nch + 1)

        ! pass 2: local exclusive scan + scatter
        !$omp parallel do
        do k = 1, nch
            a = (k - 1) * cl + 1
            b = min(k * cl, len_1d)
            b0 = base_off(k)
            lc = 0
            do i = a, b
                if (src(i) > 0.0d0) then
                    packed(b0 + lc + 1) = src(i) * weight(i)
                    lc = lc + 1
                end if
            end do
        end do
        !$omp end parallel do

    end subroutine compact_threshold_pack_fp64

end module ctp_mod
