module tsvc_2_s233_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s233_fp64")
    ! Arguments
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(inout) :: bb
    real(c_double), dimension(*), intent(in) :: cc
    integer(c_int64_t), value :: LEN_2D
    ! Locals
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx
    
    !$omp parallel private(i,j,idx)
    ! Vertical prefix sum for aa (parallel over columns i)
    !$omp do schedule(static)
    do i = 8, LEN_2D-1
      do j = 8, LEN_2D-1
        idx = j * LEN_2D + i
        aa(idx + 1) = aa(idx - LEN_2D + 1) + cc(idx + 1)
      end do
    end do
    !$omp end do
    
    ! Horizontal prefix sum for bb (parallel over rows j)
    !$omp do schedule(static)
    do j = 8, LEN_2D-1
      do i = 8, LEN_2D-1
        idx = j * LEN_2D + i
        bb(idx + 1) = bb(idx) + cc(idx + 1)
      end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine tsvc_2_s233_fp64
end module tsvc_2_s233_mod
