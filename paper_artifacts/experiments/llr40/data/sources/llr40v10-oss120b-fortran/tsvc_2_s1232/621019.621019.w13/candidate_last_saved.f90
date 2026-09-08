module tsvc_2_s1232_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
    ! Arguments:
    !   aa  - output array (double precision)
    !   bb, cc - input arrays (double precision)
    !   LEN_2D - matrix dimension (N)
    !   VLEN   - stride multiplier (positive integer)
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(in)    :: bb
    real(c_double), dimension(*), intent(in)    :: cc
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t), value :: VLEN

    integer(c_int64_t) :: i, j, idx, j_end

    ! Loop over rows i (outer). For each row, compute the maximum column index
    ! satisfying i >= (j-1)*VLEN (i.e., i >= j*VLEN in 0‑based indexing).
    !$omp parallel do default(none) shared(aa, bb, cc, LEN_2D, VLEN) private(i, j, idx, j_end)
    do i = 1, LEN_2D
       ! The largest column (1‑based) that fulfills the condition is floor((i-1)/VLEN)+1.
       j_end = (i-1) / VLEN + 1
       if (j_end > LEN_2D) j_end = LEN_2D
       !$omp simd
       do j = 1, j_end
          idx = (i-1) * LEN_2D + j   ! Row‑major linear index (1‑based)
          aa(idx) = bb(idx) + cc(idx)
       end do
    end do
    !$omp end parallel do

  end subroutine tsvc_2_s1232_fp64
end module tsvc_2_s1232_mod
