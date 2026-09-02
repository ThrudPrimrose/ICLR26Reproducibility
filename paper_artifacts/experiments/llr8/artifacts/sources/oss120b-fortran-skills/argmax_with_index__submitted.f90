module argmax_with_index_mod
  use iso_c_binding
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(C)
    ! Arguments follow C call signature: double* a, int64_t* out_index, double* out_value,
    ! int64_t LEN_1D, uint8_t* workspace, int64_t workspace_size
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(1)
    real(c_double), intent(out) :: out_value(1)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i
    real(c_double) :: max_val
    integer(c_int64_t) :: max_idx0
    real(c_double) :: local_max_val
    integer(c_int64_t) :: local_max_idx

    max_val = -huge(0.0_c_double)
    max_idx0 = -1_c_int64_t
    !$omp parallel private(i, local_max_val, local_max_idx) default(none) shared(a, LEN_1D, max_val, max_idx0)
      local_max_val = -huge(0.0_c_double)
      local_max_idx = -1_c_int64_t
      !$omp do schedule(static)
      do i = 1_c_int64_t, LEN_1D
        if (a(i) > local_max_val) then
          local_max_val = a(i)
          local_max_idx = i - 1_c_int64_t
        end if
      end do
      !$omp critical
        if (local_max_val > max_val) then
          max_val = local_max_val
          max_idx0 = local_max_idx
        end if
      !$omp end critical
    !$omp end parallel
    out_value(1) = max_val
    out_index(1) = max_idx0
  end subroutine argmax_with_index_fp64
end module argmax_with_index_mod
