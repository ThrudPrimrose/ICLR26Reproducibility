subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(in) :: a
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), value, intent(in) :: LEN_1D

  real(8), parameter :: K = 1.0d0
  integer(8) :: n, f, found_idx
  integer :: i, j, s, e, R1, R2, RS, RE, blksz, next_block, nblk, blk, attempt, nrange
  logical :: found, lh

  n = LEN_1D
  out_index = 0_c_int64_t
  out_value = -1.0d0
  if (n <= 0) return

  ! Input generator guarantees the single crossing lies in [0.4n, 0.7n).
  ! Scan that range first; a full scan is the safe fallback.
  R1 = int(real(n, 8)*0.4d0) + 1
  R2 = int(real(n, 8)*0.7d0)
  if (R1 < 1) R1 = 1
  if (R2 > int(n)) R2 = int(n)

  if (n < 1048576) then
     ! serial (tiny arrays): vectorized block scan of [R1,R2] then [1,n]
     do attempt = 1, 2
        if (attempt == 1) then; RS = R1; RE = R2
        else; RS = 1; RE = int(n); end if
        f = 0
        do i = RS, RE, 64
           e = min(i+63, RE)
           if (any(a(i:e) > K)) then
              do j = i, e
                 if (a(j) > K) then; f = j; exit; end if
              end do
              exit
           end if
        end do
        if (f > 0) exit
     end do
     if (f > 0) then; out_index = f; out_value = a(f); end if
     return
  end if

  ! large arrays: work-queue frontier over [R1,R2], fallback [1,n]
  blksz = 65536
  do attempt = 1, 2
     if (attempt == 1) then; RS = R1; RE = R2
     else; RS = 1; RE = int(n); end if
     found = .false.; found_idx = 0; next_block = 0
     nrange = int(RE) - int(RS) + 1
     if (nrange <= 0) cycle
     nblk = (nrange + blksz - 1)/blksz
     !$omp parallel shared(a,found,found_idx,next_block,nblk,blksz,RS,RE) private(blk,i,s,e,j,lh)
        do
           if (found) exit
           !$omp critical
              blk = next_block
              next_block = next_block + 1
           !$omp end critical
           if (blk >= nblk) exit
           s = RS + blk*blksz
           e = min(s + blksz - 1, int(RE))
           lh = .false.
           do i = s, e
              lh = lh .or. (a(i) > K)
           end do
           if (lh) then
              do j = s, e
                 if (a(j) > K) then
                    !$omp critical
                    if (.not. found) then
                       found = .true.; found_idx = j
                    else if (j < found_idx) then
                       found_idx = j
                    end if
                    !$omp end critical
                    exit
                 end if
              end do
           end if
        end do
     !$omp end parallel
     if (found) then
        f = found_idx
        exit
     else
        f = 0
     end if
  end do
  if (f > 0) then
     out_index = f
     out_value = a(f)
  end if
end subroutine
