module tsvc_2_s1232_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen) bind(C, name="tsvc_2_s1232_fp64")
    implicit none
    integer(c_int64_t), value :: len_2d, vlen
    real(c_double), intent(out) :: aa(*)
    real(c_double), intent(in) :: bb(*), cc(*)
    integer(c_int64_t) :: i, j, maxj
    ! Parallelize over outer loop (i). Use static scheduling.
    !$omp parallel do schedule(static) private(j, maxj) shared(aa, bb, cc, len_2d, vlen)
    do i = 0_c_int64_t, len_2d - 1_c_int64_t
      maxj = i / vlen
      !$omp simd
      do j = 0_c_int64_t, maxj
        ! Fortran arrays are 1-indexed, whereas the C reference uses 0-indexing.
        ! Adding +1 to the linear index aligns Fortran's indexing with the C code.
        aa(i * len_2d + j + 1_c_int64_t) = bb(i * len_2d + j + 1_c_int64_t) + cc(i * len_2d + j + 1_c_int64_t)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s1232_fp64
end module tsvc_2_s1232_mod
