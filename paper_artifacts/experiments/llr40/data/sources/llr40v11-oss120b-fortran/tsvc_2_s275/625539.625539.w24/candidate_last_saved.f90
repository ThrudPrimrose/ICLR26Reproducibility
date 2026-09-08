subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), dimension(*), intent(inout) :: aa
  real(c_double), dimension(*), intent(in) :: bb, cc
  integer(c_int64_t) :: i, j, p, nact
  integer(c_int64_t) :: base, prev_base
  logical, allocatable :: active(:)
  integer(c_int64_t), allocatable :: act_idx(:)

  allocate(active(LEN_2D))
  ! Determine active columns
  do i = 1, LEN_2D
    active(i) = aa(i) > 0.0_c_double
  end do

  allocate(act_idx(LEN_2D))
  nact = 0
  do i = 1, LEN_2D
    if (active(i)) then
      nact = nact + 1
      act_idx(nact) = i
    end if
  end do
  deallocate(active)

  !$omp parallel default(none) shared(act_idx, nact, aa, bb, cc, LEN_2D) private(j, base, prev_base, p, i)
  do j = 2, LEN_2D
    base = (j-1) * LEN_2D
    prev_base = base - LEN_2D
    !$omp do schedule(static)
    do p = 1, nact
      i = act_idx(p)
      aa(base + i) = aa(prev_base + i) + bb(base + i) * cc(base + i)
    end do
    !$omp end do
  end do
  !$omp end parallel

  deallocate(act_idx)

end subroutine tsvc_2_s275_fp64
