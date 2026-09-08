module tsvc_2_s231_mod
  use iso_c_binding
  implicit none
contains

subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name='tsvc_2_s231_fp64')
  ! C-ABI compliant subroutine for TSVC kernel s231 (double precision)
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double) :: aa(*)
  real(c_double) :: bb(*)
  integer(c_int64_t) :: i, j, idx
  ! Parallelize across columns (i dimension). Each column independent.
  !$omp parallel do private(j, idx) schedule(static)
  do i = 0, LEN_2D-1
    do j = 1, LEN_2D-1
      idx = j * LEN_2D + i + 1
      aa(idx) = aa(idx - LEN_2D) + bb(idx)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s231_fp64

end module tsvc_2_s231_mod
