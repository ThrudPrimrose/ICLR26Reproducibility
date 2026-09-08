! Triangular north+west wavefront: for i=1..N-1, j=i..N-1:
!   a[i,j] += a[i-1,j] + a[i,j-1]   (in place, row-major N x N, 0-based)
!
! v3: antidiagonal wavefront in (i, v=i+j) space, v = 2..2N-2.
! Cell (i, v) reads (i-1, v-1) and (i, v-1) only, so each row-owner can
! march forward in v as soon as its left neighbour has finished v-1.
! Thread t owns rows [cs(t), ce(t)] (contiguous, weight-balanced by
! N-i cells per row) plus the left halo row max(1, cs-1), which it
! recomputes redundantly (bit-identical to the neighbour's result).
! Hand-rolled sync: DONE(t) = last v completed by thread t, read with
! omp_atomic_read; no barriers, no nested teams.
subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value, intent(in) :: a
  integer(c_int64_t), value :: LEN_2D
  integer, parameter :: MAXT = 64
  integer(c_int64_t) :: N, i, j, k, vb, Wtot, tgt, acc
  integer(c_int64_t) :: cs(0:MAXT-1), ce(0:MAXT-1), ht(0:MAXT-1)
  integer :: done(0:MAXT-1), tid, tmp, NT
  integer(c_int64_t) :: imin, imax, vs, ve
  real(c_double), pointer, contiguous :: AA(:)

  N = LEN_2D
  if (N .le. 1) return
  call c_f_pointer(a, AA, [N*N])
  ! AA(i*N + j + 1) is the 0-based element a[i,j]

  if (N .lt. 1024) then
    do i = 1, N-1
      do j = i, N-1
        AA(i*N + j + 1) = AA(i*N + j + 1) + AA((i-1)*N + j + 1) + AA(i*N + j)
      end do
    end do
  else
    NT = omp_get_max_threads()
    if (NT .gt. MAXT) NT = MAXT
    if (NT .lt. 1) NT = 1
    if (NT .gt. N/2) NT = N/2

    ! split rows 1..N-1 into T contiguous weight-balanced chunks
    ! (row i holds N-i updated cells)
    Wtot = N*(N-1)/2
    acc = 0
    k = 0
    do tid = 0, NT-1
      cs(tid) = k + 1
      ce(tid) = k
      tgt = Wtot*(tid+1)/NT
      do while (k .lt. N-1 .and. acc + (N - (k+1)) .le. tgt)
        acc = acc + N - (k+1)
        k = k + 1
      end do
      ce(tid) = k
    end do
    do tid = 0, NT-1
      ht(tid) = max(1_8, cs(tid) - 1)   ! left halo row
    end do
    done = 0

    !$omp parallel shared(N, NT, cs, ce, ht, done, AA) &
    !$omp& private(tid, vb, i, imin, imax, vs, ve, tmp)
      tid = omp_get_thread_num()
      if (tid .lt. NT) then
        if (cs(tid) .le. ce(tid)) then
          vs = 2*ht(tid)
          ve = ce(tid) + N - 1
          do vb = vs, ve
            if (tid .gt. 0 .and. ht(tid) .ge. 2) then
              do
                call omp_atomic_read(done(tid-1), tmp)
                if (tmp .ge. vb-1) exit
              end do
            end if
            imin = max(ht(tid), vb - N + 1)
            imax = min(ce(tid), vb/2)
            do i = imin, imax
              AA(i*N + (vb - i) + 1) = AA(i*N + (vb - i) + 1) &
                                     + AA((i-1)*N + (vb - i) + 1) &
                                     + AA(i*N + (vb - i))
            end do
            if (tid .lt. NT-1) call omp_atomic_write(done(tid), int(vb))
          end do
        else
          call omp_atomic_write(done(tid), 2*int(N))
        end if
      end if
    !$omp end parallel
  end if
end subroutine wf_triangular_fp64
