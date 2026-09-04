module tsvc_2_s275_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
    ! Arguments: aa (inout), bb (in), cc (in) as flat double arrays, and LEN_2D dimension
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: bb(*), cc(*)
    integer(c_int64_t), value    :: LEN_2D
    integer(c_int64_t) :: i, j
    ! Parallelize over columns (i)
    !$omp parallel do private(i, j) schedule(static)
    do i = 0, LEN_2D-1
      if (aa(i+1) > 0.0_c_double) then
        do j = 1, LEN_2D-1
          aa(j*LEN_2D + i + 1) = aa((j-1)*LEN_2D + i + 1) + bb(j*LEN_2D + i + 1) * cc(j*LEN_2D + i + 1)
        end do
      end if
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
