subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*), c(*)
  integer(c_int64_t), value     :: LEN_1D

  integer(c_int64_t) :: n, i0, i1, k, len, n1
  integer, parameter :: BS = 2048
  real(c_double) :: prev(0:BS-1)
  real(c_double) :: boundary

  n = LEN_1D
  if (n <= 0) return

  ! First block is processed serially to avoid out-of-bounds warnings when
  ! computing the boundary product b(i0-1)*c(i0-1) in the parallel loop.
  n1 = min(int(BS, c_int64_t), n)
  i0 = 1
  i1 = n1
  len = i1 - i0 + 1

  !$omp simd
  do k = 0, len - 1
     a(i0 + k) = b(i0 + k) * c(i0 + k)
  end do
  !$omp end simd

  prev(0) = 0.0_c_double
  !$omp simd
  do k = 1, len - 1
     prev(k) = a(i0 + k - 1)
  end do
  !$omp end simd

  !$omp simd
  do k = 0, len - 1
     a(i0 + k) = a(i0 + k) + prev(k)
  end do
  !$omp end simd

  if (n1 >= n) return

  !$omp parallel do schedule(static) private(i0, i1, k, len, boundary, prev)
  do i0 = n1 + 1, n, int(BS, c_int64_t)
     i1 = min(i0 + BS - 1, n)
     len = i1 - i0 + 1

     ! Pass 1: products in-place (separate multiply).
     !$omp simd
     do k = 0, len - 1
        a(i0 + k) = b(i0 + k) * c(i0 + k)
     end do
     !$omp end simd

     ! Shifted previous-product buffer. Reading from a guarantees the product
     ! was already rounded, so the following add cannot be contracted into FMA.
     boundary = b(i0 - 1) * c(i0 - 1)
     prev(0) = boundary
     !$omp simd
     do k = 1, len - 1
        prev(k) = a(i0 + k - 1)
     end do
     !$omp end simd

     ! Pass 2: vector add of consecutive products.
     !$omp simd
     do k = 0, len - 1
        a(i0 + k) = a(i0 + k) + prev(k)
     end do
     !$omp end simd
  end do
  !$omp end parallel do
end subroutine tsvc_2_s252_fp64
