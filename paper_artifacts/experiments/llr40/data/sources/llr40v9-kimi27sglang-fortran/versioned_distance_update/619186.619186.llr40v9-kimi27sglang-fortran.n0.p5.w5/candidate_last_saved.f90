subroutine versioned_distance_update_fp64(a, b, c, k, len_1d) &
     bind(c, name="versioned_distance_update_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: k
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in)    :: b(len_1d)
  real(c_double), intent(in)    :: c(len_1d)

  integer(c_int64_t) :: n, kk, i, block_start, block_end, bs
  real(c_double), parameter :: d = 0.75_c_double

  n = len_1d
  kk = k
  if (n <= kk .or. kk <= 0_c_int64_t) return

  if (kk == 1_c_int64_t) then
     do i = 2_c_int64_t, n
        a(i) = d * a(i - 1_c_int64_t) + b(i) * c(i)
     end do
     return
  end if

  ! Process the array in independent contiguous blocks of size <= kk.
  ! Each element in a block depends on an index at least one block earlier,
  ! so all elements inside a block can be computed in parallel / vectorised.
  bs = kk
  if (bs > 4096_c_int64_t) bs = 4096_c_int64_t

  block_start = kk + 1_c_int64_t
  do while (block_start <= n)
     block_end = min(block_start + bs - 1_c_int64_t, n)

     if (bs >= 1024_c_int64_t) then
        !$omp parallel do simd schedule(static) private(i)
        do i = block_start, block_end
           a(i) = d * a(i - kk) + b(i) * c(i)
        end do
        !$omp end parallel do simd
     else
        !$omp simd
        do i = block_start, block_end
           a(i) = d * a(i - kk) + b(i) * c(i)
        end do
        !$omp end simd
     end if

     block_start = block_end + 1_c_int64_t
  end do

end subroutine versioned_distance_update_fp64
