module tsvc_2_s323_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(c, name='tsvc_2_s323_fp64')
    real(c_double), intent(inout) :: a(*), b(*)
    real(c_double), intent(in) :: c(*), d(*), e(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, n
    integer :: nt, tid
    integer(c_int64_t) :: chunk, start, finish
    real(c_double), allocatable :: x(:)
    real(c_double), allocatable :: offsets(:)
    real(c_double) :: acc, local_sum

    n = LEN_1D - 1
    if (n <= 0) return

    ! For small sizes, sequential is faster
    if (n < 4096) then
      do i = 2, LEN_1D
        a(i) = b(i - 1) + c(i) * d(i)
        b(i) = a(i) + c(i) * e(i)
      end do
      return
    end if

    !$omp parallel private(nt, tid, chunk, start, finish, i, local_sum, acc) shared(x, offsets)
    nt = omp_get_num_threads()
    tid = omp_get_thread_num()
    !$omp single
    allocate(x(2:LEN_1D))
    allocate(offsets(0:nt - 1))
    !$omp end single

    chunk = (n + nt - 1) / nt
    start = 2 + tid * chunk
    finish = min(start + chunk - 1, LEN_1D)

    ! Phase 1: compute increments x(i) = c(i)*(d(i)+e(i))
    if (start <= finish) then
      !$omp simd
      do i = start, finish
        x(i) = c(i) * (d(i) + e(i))
      end do
      !$omp end simd

      ! Phase 2: local sum within chunk
      local_sum = 0.0_c_double
      do i = start, finish
        local_sum = local_sum + x(i)
      end do
      offsets(tid) = local_sum
    else
      offsets(tid) = 0.0_c_double
      local_sum = 0.0_c_double
    end if

    !$omp barrier

    ! Phase 3: sequential scan of chunk totals (only one thread)
    !$omp single
    acc = b(1)
    do i = 0, nt - 1
      acc = acc + offsets(i)
      offsets(i) = acc
    end do
    !$omp end single

    !$omp barrier

    ! Phase 4: apply chunk offsets and compute a/b together
    if (start <= finish) then
      acc = offsets(tid) - local_sum
      do i = start, finish
        a(i) = acc + c(i) * d(i)
        acc = acc + x(i)
        b(i) = acc
      end do
    end if
    !$omp end parallel
  end subroutine tsvc_2_s323_fp64
end module tsvc_2_s323_mod
