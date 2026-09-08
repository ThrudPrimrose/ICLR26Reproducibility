module ext_break_capture_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(*)
    real(c_double), intent(out) :: out_value(*)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: idx
    integer(c_int64_t) :: local_min
    real(c_double), parameter :: K = 1.0_c_double
    ! Initialize sentinel values as in the C reference
    out_index(1) = -1_c_int64_t
    out_value(1) = -1.0_c_double
    idx = LEN_1D
    !$omp parallel if (LEN_1D > 1000) private(local_min)
    local_min = LEN_1D
    !$omp do schedule(static)
    do i = 1, LEN_1D
      if (a(i) > K) then
        if (i - 1_c_int64_t < local_min) local_min = i - 1_c_int64_t
      end if
    end do
    !$omp end do
    !$omp critical
      if (local_min < idx) idx = local_min
    !$omp end critical
!$omp end parallel
    if (idx < LEN_1D) then
      out_index(1) = idx
      out_value(1) = a(idx + 1)
    end if
  end subroutine ext_break_capture_fp64
end module ext_break_capture_mod
