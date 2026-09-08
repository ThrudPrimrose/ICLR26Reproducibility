subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  integer(c_int64_t) :: i
  real(c_double) :: maxv
  integer(c_int64_t) :: idx
  integer(c_int64_t) :: sentinel

  ! First reduction: find the maximum value
  maxv = -huge(0.0_c_double)
  !$omp parallel do reduction(max:maxv) schedule(static)
  do i = 1, LEN_1D
    if (a(i) > maxv) then
      maxv = a(i)
    end if
  end do
  !$omp end parallel do

  ! Second reduction: find the first index (1‑based) where the maximum occurs
  sentinel = LEN_1D + 1_c_int64_t
  idx = sentinel
  !$omp parallel do reduction(min:idx) schedule(static)
  do i = 1, LEN_1D
    if (a(i) == maxv) then
      idx = i
    else
      idx = sentinel
    end if
  end do
  !$omp end parallel do

  out_value(1) = maxv
  out_index(1) = idx
end subroutine argmax_with_index_fp64
