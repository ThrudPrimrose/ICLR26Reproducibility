module tsvc_2_s232_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s232_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s232_fp64")
    ! Arguments: aa (inout), bb (in), LEN_2D (size of each dimension)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: bb(*)
    integer(c_int64_t), value    :: LEN_2D
    integer(c_int64_t) :: j, i, base, offset, offset_im1

    !$omp parallel do default(none) shared(aa, bb, LEN_2D) private(j,i,base,offset,offset_im1)
    do j = 1, LEN_2D - 1_c_int64_t
      base = j * LEN_2D
      do i = 1, j
        offset = base + i
        offset_im1 = base + (i - 1_c_int64_t)
        aa(offset + 1) = aa(offset_im1 + 1) * aa(offset_im1 + 1) + bb(offset + 1)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s232_fp64
end module tsvc_2_s232_mod
