! TSVC s481 (ext_break_find_first):
!   a[i] = a[i] + b[i]*c[i]  for i = 0..LEN-1, stopping (break) at the first i with d[i] < 0.
! C-ABI entry: void ext_break_find_first(double* a, const double* b,
!                                        const double* c, const double* d, long len1d)
! Strategy:
!   phase 1: multi-threaded chunked scan of d for the first element < 0.
!            Each chunk is probed in 16-wide blocks (vector minval) with a scalar
!            confirm; the first finder publishes the index (critical, rare path) and
!            chunks whose start lies after the published index stop immediately.
!   phase 2: single in-place vectorized (AVX-512) multi-threaded update over [0, ncut).

! C-ABI entry required by the harness:
!   void ext_break_find_first_fp64(double *a, double *b, double *c, double *d,
!                                  int64_t len1d, uint8_t *ws, int64_t wsbytes)
      subroutine ext_break_find_first_fp64(a, b, c, d, len1d, ws, wsbytes) bind(c)
        use iso_c_binding, only: c_ptr, c_null_ptr, c_f_pointer
        implicit none
        type(c_ptr), value, intent(in) :: a, b, c, d, ws
        integer(8), value, intent(in) :: len1d
        integer(8), value, intent(in) :: wsbytes
        real(8), pointer :: at(:), bt(:), ct(:), dt(:)
        integer(8) :: n
        n = len1d
        if (n > 0) then
          call c_f_pointer(a, at, [n])
          call c_f_pointer(b, bt, [n])
          call c_f_pointer(c, ct, [n])
          call c_f_pointer(d, dt, [n])
          call ebff_run(at, bt, ct, dt, n)
        end if
      end subroutine ext_break_find_first_fp64

! ------------------------------------------------------------------
      subroutine ebff_run(a, b, c, d, n)
        use omp_lib
        implicit none
        integer, parameter :: ik8 = selected_int_kind(15)
        real(8), parameter :: z8 = 0.0d0
        integer, parameter :: BLK = 16
        integer(ik8), intent(in)    :: n
        real(8), intent(inout)      :: a(n)
        real(8), intent(in)         :: b(n)
        real(8), intent(in)         :: c(n)
        real(8), intent(in)         :: d(n)
        integer(ik8) :: ncut, gfirst, it, nt, chunk, nchunks
        integer(ik8) :: lo, hi, i, j, jj
        real(8) :: mn

        ! ---------- phase 1: ncut = number of updated elements ----------
        ! (count of elements strictly before the first d < 0)
        if (n < 2000000_ik8) then
          ncut = n
          do i = 1, n
            if (d(i) < z8) then
              ncut = i - 1
              exit
            end if
          end do
        else
          nt = max(int(1, kind=ik8), int(omp_get_max_threads(), kind=ik8))
          chunk = (n + nt - 1) / nt
          nchunks = (n + chunk - 1) / chunk
          gfirst = n + 1
          !$omp parallel do schedule(static) private(lo, hi, j, jj, mn)
          do it = 1, nchunks
            lo = (it - 1) * chunk + 1
            hi = min(lo + chunk - 1, n)
            j = lo
            do while (j + BLK - 1 <= hi)
              if (gfirst <= lo) exit
              mn = minval(d(j:j+BLK-1))
              if (mn < z8) then
                do jj = j, j+BLK-1
                  if (d(jj) < z8) then
                    !$omp critical
                    if (jj < gfirst) gfirst = jj
                    !$omp end critical
                    exit
                  end if
                end do
                exit
              end if
              j = j + BLK
            end do
            do while (j <= hi)
              if (gfirst <= lo) exit
              if (d(j) < z8) then
                !$omp critical
                if (j < gfirst) gfirst = j
                !$omp end critical
                exit
              end if
              j = j + 1
            end do
          end do
          !$omp end parallel do
          ncut = gfirst - 1
          if (ncut > n) ncut = n
        end if

        if (ncut <= 0) return

        ! ---------- phase 2: in-place vectorized update ----------
        if (ncut < 2000000_ik8) then
          !$omp simd
          do i = 1, ncut
            a(i) = a(i) + b(i) * c(i)
          end do
        else
          !$omp parallel do simd schedule(static)
          do i = 1, ncut
            a(i) = a(i) + b(i) * c(i)
          end do
        end if
      end subroutine ebff_run
