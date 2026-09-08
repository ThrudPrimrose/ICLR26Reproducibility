module compact_threshold_pack_mod
  use iso_c_binding
  implicit none
contains
  subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, LEN_1D) bind(C)
    ! Arguments
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: src(LEN_1D), weight(LEN_1D)
    real(c_double), intent(out) :: packed(LEN_1D)
    integer(c_int64_t), intent(out) :: out_count(1)
    integer :: i
    integer(c_int64_t) :: n
    real(c_double) :: tmp
    n = 0_c_int64_t
    !$omp parallel
    !$omp master
    i = 1
    do while (i <= LEN_1D)
      if (src(i) > 0.0_c_double) then
        packed(n+1) = src(i) * weight(i)
        n = n + 1_c_int64_t
      end if
      i = i + 1
    end do
    !$omp end master
    !$omp end parallel
    out_count(1) = n
  end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_mod
