subroutine tsvc_2_s318_fp64(a, result, len_1d, inc) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, inc
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)

  integer(c_int64_t) :: n, i, j, b, k, i_lo, i_hi, ks, ke, lc, full_end, last, idx
  integer :: nt, t
  real(c_double) :: M, pmv
  logical :: hit
  real(c_double), allocatable :: pm(:)
  integer(c_int64_t), allocatable :: cand(:)

  n = len_1d
  if (n <= 0) then
    result(1) = 0.0d0
    return
  end if

  nt = omp_get_max_threads()
  allocate (pm(1:nt), cand(1:nt))

  !$omp parallel
    t = omp_get_thread_num() + 1
    i_lo = (n * (t - 1)) / nt
    i_hi = (n * t) / nt - 1

    pmv = -HUGE(1.0d0)
    if (inc == 1) then
      !$omp simd reduction(max:pmv)
      do k = i_lo + 1, i_hi + 1
        pmv = max(pmv, abs(a(k)))
      end do
    else
      !$omp simd reduction(max:pmv)
      do i = i_lo, i_hi
        pmv = max(pmv, abs(a(1 + i * inc)))
      end do
    end if
    pm(t) = pmv
  !$omp barrier
    M = maxval(pm)
    cand(t) = n
    if (inc == 1) then
      if (pm(t) == M) then
        hit = .false.
        ks = i_lo + 1
        ke = i_hi + 1
        lc = ke - ks + 1
        if (lc >= 16) then
          full_end = ks + 16 * (lc / 16) - 1
          do b = ks, full_end - 15, 16
            if (any(abs(a(b:b+15)) == M)) then
              do j = b, b + 15
                if (abs(a(j)) == M) then
                  cand(t) = j - 1
                  hit = .true.
                  exit
                end if
              end do
              exit
            end if
          end do
          last = full_end + 1
        else
          last = ks
        end if
        if (.not. hit) then
          do j = last, ke
            if (abs(a(j)) == M) then
              cand(t) = j - 1
              hit = .true.
              exit
            end if
          end do
        end if
      end if
    else
      if (pm(t) == M) then
        hit = .false.
        do i = i_lo, i_hi
          if (abs(a(1 + i * inc)) == M) then
            cand(t) = i
            hit = .true.
            exit
          end if
        end do
      end if
    end if
  !$omp end parallel

  idx = minval(cand)
  result(1) = M + dble(idx)
end subroutine tsvc_2_s318_fp64
