subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none

  integer(c_int64_t), value, intent(in) :: K, LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size

  real(c_double), parameter :: alpha = 0.75d0
  integer(c_int64_t) :: n, kk, i, r, g, j, ng, nt, nt_use, tid, chunk, lo, hi, t
  real(c_double) :: local_sum, local_pow, carry, run
  real(c_double), allocatable :: agg_sum(:), agg_pow(:)

  n = LEN_1D
  kk = K

  if (kk <= 0 .or. n <= kk) return

  ! For small inputs the OpenMP startup cost dominates: run the recurrence
  ! serially.  Process whole groups of K consecutive elements so the
  ! inner loop is unit-stride and vectorises for kk > 1.
  if (n <= 8192) then
    if (kk == 1) then
      do i = 2, n
        a(i) = alpha * a(i - 1) + b(i) * c(i)
      end do
    else
      ng = n / kk
      do g = 2, ng
        do j = 1, kk
          i = (g - 1) * kk + j
          a(i) = alpha * a(i - kk) + b(i) * c(i)
        end do
      end do
      do i = ng * kk + 1, n
        a(i) = alpha * a(i - kk) + b(i) * c(i)
      end do
    end if
    return
  end if

  if (kk == 1) then
    nt = omp_get_max_threads()
    allocate(agg_sum(0:nt - 1), agg_pow(0:nt - 1))

    ! Update indices 2..n; a(1) is the seed.
    chunk = (n - 1 + nt - 1) / nt

    !$omp parallel private(tid, lo, hi, i, local_sum, local_pow, carry, run)
    tid = omp_get_thread_num()
    lo = tid * chunk + 2
    hi = min((tid + 1) * chunk + 1, n)

    local_sum = 0.0d0
    local_pow = 1.0d0
    if (lo <= hi) then
      do i = lo, hi
        local_sum = alpha * local_sum + b(i) * c(i)
        local_pow = local_pow * alpha
      end do
    end if
    agg_sum(tid) = local_sum
    agg_pow(tid) = local_pow

    !$omp barrier
    !$omp single
    agg_sum(0) = agg_pow(0) * a(1) + agg_sum(0)
    do t = 1, nt - 1
      agg_sum(t) = agg_pow(t) * agg_sum(t - 1) + agg_sum(t)
    end do
    !$omp end single

    if (tid == 0) then
      carry = a(1)
    else
      carry = agg_sum(tid - 1)
    end if

    run = carry
    if (lo <= hi) then
      do i = lo, hi
        run = alpha * run + b(i) * c(i)
        a(i) = run
      end do
    end if
    !$omp end parallel

    deallocate(agg_sum, agg_pow)
  else
    ! For small K there are only K independent chains; spawning more threads
    ! than chains just adds overhead.
    nt_use = min(omp_get_max_threads(), int(kk))
    !$omp parallel do simd schedule(static) num_threads(nt_use) private(i, run)
    do r = 1, kk
      if (r + kk <= n) then
        run = alpha * a(r) + b(r + kk) * c(r + kk)
        a(r + kk) = run
        do i = r + 2 * kk, n, kk
          run = alpha * a(i - kk) + b(i) * c(i)
          a(i) = run
        end do
      end if
    end do
    !$omp end parallel do simd
  end if

end subroutine versioned_distance_update_fp64
