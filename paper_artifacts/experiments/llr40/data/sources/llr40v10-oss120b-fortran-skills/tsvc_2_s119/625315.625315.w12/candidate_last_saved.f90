subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: t, i, i_low, i_high, j
  integer(c_int64_t), parameter :: two = 2_c_int64_t

  !$omp parallel private(t, i, i_low, i_high, j)
  do t = 4_c_int64_t, 2_c_int64_t * LEN_2D
    i_low = max(two, t - LEN_2D)
    i_high = min(LEN_2D, t - two)
    if (i_low <= i_high) then
      !$omp do simd schedule(static) nowait
      do i = i_low, i_high
        j = t - i
        aa(i, j) = aa(i-1, j-1) + bb(i, j)
      end do
      !$omp end do simd
    end if
  end do
  !$omp end parallel

end subroutine tsvc_2_s119_fp64
