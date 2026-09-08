module tsvc_2_s1232_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
    ! Arguments
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t), value :: VLEN
    real(c_double), intent(out) :: aa(*)
    real(c_double), intent(in) :: bb(*)
    real(c_double), intent(in) :: cc(*)
    ! Locals
    integer(c_int64_t) :: i, j, idx, max_j
    ! Parallelize outer loop over rows (i)
    !$omp parallel do schedule(static) private(i, j, idx, max_j) shared(aa, bb, cc, LEN_2D, VLEN)
    do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      if (VLEN > 0_c_int64_t) then
        max_j = i / VLEN
        if (max_j > LEN_2D - 1_c_int64_t) max_j = LEN_2D - 1_c_int64_t
        if (max_j >= 0_c_int64_t) then
          do j = 0_c_int64_t, max_j
            idx = i * LEN_2D + j
            aa(idx + 1) = bb(idx + 1) + cc(idx + 1)
          end do
        end if
      end if
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s1232_fp64
end module tsvc_2_s1232_mod
