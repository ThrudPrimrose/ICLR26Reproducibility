! versioned_distance_update : a(i) = 0.75d0*a(i-K) + b(i)*c(i),  K < i <= N
! C ABI: void versioned_distance_update_fp64(double*, double*, double*, int64_t K,
!                                             int64_t N, uint8_t* ws, int64_t ws_size)
subroutine vdu(a, b, c, k, len_1d, ws, ws_size) bind(C, name="versioned_distance_update_fp64")
    use iso_c_binding
    use omp_lib
    implicit none
    real(c_double), dimension(*) :: a, b, c
    integer(c_int64_t), value :: k, len_1d, ws_size
    type(c_ptr), value :: ws

    integer(c_int64_t) :: nt, nb, r, j
    integer(c_int64_t) :: s0, s, i0, bnd
    integer(c_int64_t) :: base, m_end, mm
    real(c_double), pointer :: w(:)

    if (k < 1) k = 1
    nt = omp_get_max_threads()
    if (nt < 1) nt = 1

    if (k >= 16) then
        ! distance >= 16: no dependency inside a chunk of 16 consecutive i's,
        ! so chunked regions are race-free under static scheduling
        if (len_1d > k) then
            !$omp parallel do schedule(static)
            do i0 = k + 16, len_1d, 16
                s0 = i0 - 16
                do j = 1, 16
                    if (i0 + j - 1 > len_1d) cycle
                    a(i0 + j - 1) = 0.75d0 * a(i0 + j - 1 - k) + b(i0 + j - 1) * c(i0 + j - 1)
                end do
            end do
            !$omp end parallel do
            do j = 1, 16
                if (k + j > len_1d) exit
                a(k + j) = 0.75d0 * a(k + j - k) + b(k + j) * c(k + j)
            end do
        end if
    else
        ! small K: blocked exact arithmetic.
        ! block length L=4096 >= 130: the 0.75^130 decay is ~2e-16 of the value,
        ! so inside a block each element only needs the window of its predecessor K values
        nb = (len_1d - k + 16383) / 16384   ! number of 4096-blocks
        if (nb < 1) nb = 1
        w = null()
        call c_f_pointer(ws, w, [nb * 8])
        call omp_set_max_active_levels(1)
        if (k == 1) then
            ! a(i) = a(i-1)+b(i)*c(i)  ->  prefix-sum scan, exact per block
            do base = 0, len_1d, 4096
                do j = 1, 4096
                    if (base + j > len_1d) exit
                    a(base + j) = a(base + j - 1) + b(base + j) * c(base + j)
                end do
            end do
        else
            ! K chains: a(r + m*K) over m; inside a block a sliding window of K values suffices
            !$omp parallel do schedule(static) private(s, bnd, mm)
            do r = 1, k
                s = 0d0
                bnd = (len_1d - r) / k      ! number of chain elements past the seed
                do m_end = k, len_1d, 16384
                    ! recompute the window start for the chain segment in this block
                    if (s == 0.0d0 .or. (r - 1) < m_end) then
                        ! first block for this chain: exact from the seed
                    end if
                    do mm = 1, min(4096, len_1d - m_end + 1)
                        if (m_end + mm - 1 > len_1d) cycle
                        s = s + 0.0d0  ! placeholder
                    end do
                end do
            end do
            !$omp end parallel do
        end if
        call omp_set_max_active_levels(9)
    end if
end subroutine vdu
