! Stream compaction (parallel, two-pass block scan, single OpenMP region).
! Pack src[i]*weight[i] for every src[i] > 0, preserve source order, publish count.
! Canonical C-ABI (c-abi-v2) entry point; trailing workspace pair is ignored.
subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_size) bind(C, &
&name="compact_threshold_pack_fp64")
    use, intrinsic :: iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), intent(inout) :: out_count(1)
    real(c_double), intent(inout) :: packed(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    real(c_double), intent(in) :: weight(LEN_1D)
    type(c_ptr), value :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size

    integer(c_int64_t) :: N, nb, b, i, lo, hi, tot, cnt
    integer :: nt
    integer(c_int64_t) :: bcnt(1024), bbase(1024)

    N = LEN_1D

    ! ---- small-N serial path (no OpenMP overhead) ----
    if (N < 131072) then
        tot = 0
        do i = 0, N - 1
            if (src(i + 1) > 0.0d0) then
                packed(tot + 1) = src(i + 1) * weight(i + 1)
                tot = tot + 1
            end if
        end do
        out_count(1) = tot
        return
    end if

    ! ---- large-N two-pass block scan, one parallel region ----
    nt = omp_get_max_threads()
    if (nt < 1) nt = 1
    nb = 8 * int(nt, c_int64_t)
    if (nb < 64) nb = 64
    if (nb > 1024) nb = 1024

    !$omp parallel
    ! Pass 1: per-block survivor counts.
    !$omp do schedule(static)
    do b = 0, nb - 1
        lo = b * (N + nb - 1) / nb
        hi = (b + 1) * (N + nb - 1) / nb
        cnt = 0
        do i = lo, hi - 1
            cnt = cnt + merge(1, 0, src(i + 1) > 0.0d0)
        end do
        bcnt(b + 1) = cnt
    end do
    !$omp end do

    ! Exclusive scan of the (small) block counts, then fan out.
    !$omp single
    tot = 0
    do b = 1, nb
        bbase(b) = tot
        tot = tot + bcnt(b)
    end do
    out_count(1) = tot
    !$omp end single

    ! Pass 2: scatter survivors into packed at base+running.
    !$omp do schedule(static)
    do b = 1, nb
        lo = (b - 1) * (N + nb - 1) / nb
        hi = b * (N + nb - 1) / nb
        cnt = bbase(b)
        do i = lo, hi - 1
            if (src(i + 1) > 0.0d0) then
                packed(cnt + 1) = src(i + 1) * weight(i + 1)
                cnt = cnt + 1
            end if
        end do
    end do
    !$omp end do
    !$omp end parallel
end subroutine compact_threshold_pack_fp64
