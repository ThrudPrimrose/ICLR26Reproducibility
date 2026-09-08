subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_bytes) bind(c, name="versioned_distance_update_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, K, workspace_bytes
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_bytes)
  integer(c_int64_t) :: i
  real(c_double) :: t, t1, t2, t3, t4, t5

  if (K <= 0 .or. LEN_1D <= K) return

  if (K == 1_c_int64_t) then
    t = a(1)
    do i = 2, LEN_1D
      t = 0.75d0 * t + b(i) * c(i)
      a(i) = t
    end do
  else if (K == 5_c_int64_t) then
    t1 = a(1)
    t2 = a(2)
    t3 = a(3)
    t4 = a(4)
    t5 = a(5)
    i = 6
    do while (i + 4 <= LEN_1D)
      t1 = 0.75d0 * t1 + b(i)   * c(i);   a(i)   = t1; i = i + 1
      t2 = 0.75d0 * t2 + b(i) * c(i);     a(i) = t2;   i = i + 1
      t3 = 0.75d0 * t3 + b(i) * c(i);     a(i) = t3;   i = i + 1
      t4 = 0.75d0 * t4 + b(i) * c(i);     a(i) = t4;   i = i + 1
      t5 = 0.75d0 * t5 + b(i) * c(i);     a(i) = t5;   i = i + 1
    end do
    do while (i <= LEN_1D)
      a(i) = 0.75d0 * a(i - 5) + b(i) * c(i)
      i = i + 1
    end do
  else if (K >= 8_c_int64_t) then
    !$omp simd simdlen(8)
    do i = K + 1, LEN_1D
      a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
    end do
  else if (K >= 4_c_int64_t) then
    !$omp simd simdlen(4)
    do i = K + 1, LEN_1D
      a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
    end do
  else
    do i = K + 1, LEN_1D
      a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
    end do
  end if
end subroutine versioned_distance_update_fp64
