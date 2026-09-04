module tsvc_2_s233_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s233_fp64")
    import :: C_DOUBLE, C_INT64_T
    real(C_DOUBLE), intent(inout) :: aa(*)
    real(C_DOUBLE), intent(inout) :: bb(*)
    real(C_DOUBLE), intent(in)    :: cc(*)
    integer(C_INT64_T), value :: LEN_2D
    integer(C_INT64_T) :: i, j
    ! Parallelize outer loop for aa across i (columns)
    !$omp parallel default(shared) private(i, j)
    !$omp do schedule(static)
    do i = 8, LEN_2D-1
      do j = 8, LEN_2D-1
        aa(j*LEN_2D + i + 1) = aa((j-1)*LEN_2D + i + 1) + cc(j*LEN_2D + i + 1)
      end do
    end do
    !$omp end do
    ! Second loop: sequential in i, but inner j loop can be parallelized
    do i = 8, LEN_2D-1
      !$omp do schedule(static) nowait
      do j = 8, LEN_2D-1
        bb(j*LEN_2D + i + 1) = bb(j*LEN_2D + (i-1) + 1) + cc(j*LEN_2D + i + 1)
      end do
      !$omp end do
    end do
    !$omp end parallel
  end subroutine tsvc_2_s233_fp64
end module tsvc_2_s233_mod
