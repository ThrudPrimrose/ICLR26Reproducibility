module tsvc_2_s275_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
    ! Arguments: double precision arrays aa (inout), bb, cc (in), LEN_2D (length of dimension)
    real(c_double), intent(inout), dimension(*) :: aa
    real(c_double), intent(in),    dimension(*) :: bb, cc
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx0, idx1

    !$omp parallel do default(none) shared(aa,bb,cc,LEN_2D) private(i,j,idx0,idx1)
    do i = 0_c_int64_t, LEN_2D-1_c_int64_t
      if (aa(i+1) > 0.0_c_double) then
        do j = 1_c_int64_t, LEN_2D-1_c_int64_t
          idx0 = (j-1_c_int64_t) * LEN_2D + i
          idx1 = j * LEN_2D + i
          aa(idx1+1) = aa(idx0+1) + bb(idx1+1) * cc(idx1+1)
        end do
      end if
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s275_fp64
end module tsvc_2_s275_mod
