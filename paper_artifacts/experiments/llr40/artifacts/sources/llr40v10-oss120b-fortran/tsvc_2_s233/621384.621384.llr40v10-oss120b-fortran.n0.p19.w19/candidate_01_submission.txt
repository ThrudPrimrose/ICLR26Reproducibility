module tsvc_2_s233_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s233_fp64")
    ! Arguments
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(inout) :: bb(*)
    real(c_double), intent(in) :: cc(*)
    ! Locals
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: stride
    stride = LEN_2D
    ! Compute AA: each column i is independent, parallelize outer loop over i.
    !$omp parallel do private(i, j) schedule(static)
    do i = 8, LEN_2D-1
      do j = 8, LEN_2D-1
        aa(j*stride + i + 1) = aa((j-1)*stride + i + 1) + cc(j*stride + i + 1)
      end do
    end do
    !$omp end parallel do
    ! Compute BB: each row j is independent, parallelize outer loop over j.
    !$omp parallel do private(i, j) schedule(static)
    do j = 8, LEN_2D-1
      do i = 8, LEN_2D-1
        bb(j*stride + i + 1) = bb(j*stride + (i-1) + 1) + cc(j*stride + i + 1)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s233_fp64
end module tsvc_2_s233_mod
