subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(*), b(*)
  real(c_double), intent(in) :: c(*), d(*), e(*)
  integer(c_int64_t) :: i, tid, t, nt
  real(c_double) :: offset, term
  real(c_double), allocatable :: partial_sums(:), offsets(:)

  if (LEN_1D <= 1_c_int64_t) return

  offset = b(1)
  nt = omp_get_max_threads()
  allocate(partial_sums(nt))
  allocate(offsets(nt))

  ! First parallel region: compute per-thread prefix sums of term and store temporary b(i)
  !$omp parallel private(i, term, tid)
    tid = omp_get_thread_num() + 1  ! Fortran 1-based index
    term = 0.0_c_double
    !$omp do schedule(static)
    do i = 2, LEN_1D
      term = term + c(i) * (d(i) + e(i))
      b(i) = term
    end do
    !$omp end do
    partial_sums(tid) = term
  !$omp end parallel

  ! Compute offsets for each thread
  offsets(1) = 0.0_c_double
  do t = 2, nt
    offsets(t) = offsets(t-1) + partial_sums(t-1)
  end do

  ! Second parallel region: add offsets and compute final a and b
  !$omp parallel private(i, tid)
    tid = omp_get_thread_num() + 1
    !$omp do schedule(static)
    do i = 2, LEN_1D
      b(i) = offset + offsets(tid) + b(i)
      a(i) = b(i) - c(i) * e(i)
    end do
    !$omp end do
  !$omp end parallel

  deallocate(partial_sums)
  deallocate(offsets)

end subroutine tsvc_2_s323_fp64
