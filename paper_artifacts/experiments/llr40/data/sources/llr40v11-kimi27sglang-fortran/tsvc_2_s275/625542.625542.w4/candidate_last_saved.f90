subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(c, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t), parameter :: NB = 64
  integer(c_int64_t) :: i, j, ib, i_block, i_end, nb_active
  real(c_double) :: prev(NB)
  logical :: active(NB)

  !$omp parallel do private(i, j, ib, i_block, i_end, nb_active, prev, active) schedule(static)
  do i_block = 1, LEN_2D, NB
    i_end = min(LEN_2D, i_block + NB - 1)
    nb_active = i_end - i_block + 1
    do ib = 1, nb_active
      prev(ib) = aa(i_block + ib - 1, 1)
      active(ib) = (prev(ib) > 0.0_c_double)
    end do
    do j = 2, LEN_2D
      !$omp simd
      do ib = 1, nb_active
        prev(ib) = prev(ib) + bb(i_block + ib - 1, j) * cc(i_block + ib - 1, j)
        aa(i_block + ib - 1, j) = merge(prev(ib), aa(i_block + ib - 1, j), active(ib))
      end do
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s275_fp64
