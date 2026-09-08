module tsvc_2_s2233_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s2233_fp64")
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(inout) :: bb(*)
    real(c_double), intent(in)    :: cc(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx, idx_prev
    ! Parallelize the column-wise update of aa (independent across i)
    !$omp parallel default(none) shared(aa, cc, LEN_2D) private(i, j, idx, idx_prev)
    !$omp do schedule(static)
    do i = 8_c_int64_t, LEN_2D - 1_c_int64_t
      do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
        idx = j * LEN_2D + i
        idx_prev = (j - 1) * LEN_2D + i
        aa(idx + 1) = aa(idx_prev + 1) + cc(idx + 1)
      end do
    end do
    !$omp end do
    !$omp end parallel
    ! Sequential row-wise update of bb (depends on previous row)
    do i = 8_c_int64_t, LEN_2D - 1_c_int64_t
      do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
        idx = i * LEN_2D + j
        idx_prev = (i - 1) * LEN_2D + j
        bb(idx + 1) = bb(idx_prev + 1) + cc(idx + 1)
      end do
    end do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233_mod
