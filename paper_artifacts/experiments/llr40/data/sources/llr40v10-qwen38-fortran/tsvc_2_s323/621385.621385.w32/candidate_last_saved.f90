subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in)    :: LEN_1D
  real(c_double), intent(inout)     :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in)        :: c(LEN_1D), d(LEN_1D), e(LEN_1D)

  real(c_double) :: b1, run, base
  real(c_double), allocatable :: T(:), offs(:)
  integer(c_int64_t) :: N, M, S, NB, j, k, lo, hi

  N = LEN_1D
  M = N - 1
  if (M < 1) return
  b1 = b(1)

  S  = 8192
  NB = (M + S - 1) / S
  allocate(T(NB), offs(NB))

  ! Phase 1: local inclusive prefix of w(j)=c(j)*d(j)+c(j)*e(j) within each block,
  ! stored in-place into b(j). Serial scan inside block (exact order).
  !$omp parallel do schedule(static)
  do k = 0, NB-1
    lo = 2 + k*S
    hi = min((k+1)*S + 1, N)
    run = 0.0d0
    do j = lo, hi
      run = run + (c(j)*d(j) + c(j)*e(j))
      b(j) = run
    end do
    T(k) = run
  end do

  ! Phase 2: serial prefix of block totals (tiny).
  base = 0.0d0
  do k = 0, NB-1
    offs(k) = base
    base = base + T(k)
  end do

  ! Phase 3: apply offset, finalize b and a. Element-wise -> vectorized, parallel.
  !$omp parallel do schedule(static)
  do k = 0, NB-1
    lo = 2 + k*S
    hi = min((k+1)*S + 1, N)
    base = offs(k)
    do j = lo, hi
      b(j) = b1 + base + b(j)
      a(j) = b(j) - c(j)*e(j)
    end do
  end do

  deallocate(T, offs)
end subroutine tsvc_2_s323_fp64
