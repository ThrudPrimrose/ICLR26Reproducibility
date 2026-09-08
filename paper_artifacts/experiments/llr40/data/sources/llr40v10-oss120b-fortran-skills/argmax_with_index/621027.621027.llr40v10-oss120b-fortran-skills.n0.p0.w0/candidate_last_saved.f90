module argmax_with_index_mod
  use iso_c_binding
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(C)
    ! Arguments: a - input array, out_value/out_index - scalar outputs, LEN_1D - length
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: a(LEN_1D)
    integer(c_int64_t), intent(out) :: out_index(1)
    real(c_double), intent(out) :: out_value(1)
    real(c_double) :: max_val
    integer(c_int64_t) :: i

    if (LEN_1D <= 0_c_int64_t) then
      out_value(1) = 0.0_c_double
      out_index(1) = -1_c_int64_t
      return
    end if

    max_val = a(1)
    if (LEN_1D > 1_c_int64_t) then
      !$omp parallel do reduction(max:max_val) default(none) shared(a, LEN_1D) private(i) schedule(static)
      do i = 2_c_int64_t, LEN_1D
        if (a(i) > max_val) max_val = a(i)
      end do
      !$omp end parallel do
    end if

    out_value(1) = max_val
    out_index(1) = -1_c_int64_t
    do i = 1_c_int64_t, LEN_1D
      if (a(i) == max_val) then
        out_index(1) = i
        exit
      end if
    end do
  end subroutine argmax_with_index_fp64
end module argmax_with_index_mod
