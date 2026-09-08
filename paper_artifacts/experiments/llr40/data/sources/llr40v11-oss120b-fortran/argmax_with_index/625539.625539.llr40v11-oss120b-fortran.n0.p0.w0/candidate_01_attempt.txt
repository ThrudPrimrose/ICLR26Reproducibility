subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), value :: LEN_1D
  real(c_double) :: first_val, max_rest, out_val
  integer(c_int64_t) :: idx_first, idx_rest, out_idx
  integer(c_int64_t) :: i
  real(c_double) :: local_max
  integer(c_int64_t) :: local_idx

  first_val = a(1)
  idx_first = 0_c_int64_t
  max_rest = -huge(1.0_c_double)
  idx_rest = -1_c_int64_t

  if (LEN_1D > 1_c_int64_t) then
    !$omp parallel private(i, local_max, local_idx)
      local_max = -huge(1.0_c_double)
      local_idx = -1_c_int64_t
      !$omp do nowait schedule(static)
      do i = 2, LEN_1D
        if (a(i) > local_max) then
          local_max = a(i)
          local_idx = i - 1_c_int64_t
        end if
      end do
      !$omp critical
        if (local_max > max_rest) then
          max_rest = local_max
          idx_rest = local_idx
        end if
      !$omp end critical
    !$omp end parallel
  end if

  if (ieee_is_nan(first_val)) then
    out_val = first_val
    out_idx = idx_first
  else if (first_val > max_rest) then
    out_val = first_val
    out_idx = idx_first
  else
    out_val = max_rest
    out_idx = idx_rest
  end if

  out_value = out_val
  out_index = out_idx
end subroutine argmax_with_index_fp64
